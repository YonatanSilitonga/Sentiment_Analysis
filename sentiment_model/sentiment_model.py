from __future__ import annotations

import math
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from sentiment_model.long_text import classify_long_text, split_long_text_segments
    from sentiment_model.preprocessing import IndonesianTextPreprocessor, LABEL_ORDER
    from sentiment_model.test_model import load_artifacts, predict_texts
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sentiment_model.long_text import classify_long_text, split_long_text_segments
    from sentiment_model.preprocessing import IndonesianTextPreprocessor, LABEL_ORDER
    from sentiment_model.test_model import load_artifacts, predict_texts


INTERNAL_TO_EXTERNAL_LABEL = {
    "negatif": "negative",
    "netral": "neutral",
    "positif": "positive",
}

EXTERNAL_TO_INTERNAL_LABEL = {value: key for key, value in INTERNAL_TO_EXTERNAL_LABEL.items()}


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _clean_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    if isinstance(value, tuple):
        return [_clean_value(item) for item in value]
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (str, bytes, int, float, bool)):
        return value
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.to_dict()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _external_label(label: str) -> str:
    return INTERNAL_TO_EXTERNAL_LABEL.get(label, label)


def _internal_label(label: str) -> str:
    return EXTERNAL_TO_INTERNAL_LABEL.get(label, label)


def _scores_from_row(row: pd.Series, label_order: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label in label_order:
        external = _external_label(label)
        scores[external] = float(row.get(f"prob_{label}", 0.0) or 0.0)
    return scores


def _align_label_and_confidence(label: str, scores: dict[str, float], fallback_confidence: float = 0.0) -> tuple[str, float]:
    if label in scores:
        return label, float(scores[label])
    if scores:
        best_label = max(scores, key=scores.get)
        return str(best_label), float(scores[best_label])
    return label, float(fallback_confidence)


def _is_close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def _first_non_empty_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return "model_only"

@dataclass
class SentimentPrediction:
    label: str
    confidence: float
    scores: dict[str, float]
    reason: str
    processed_text: str
    review_id: Any | None = None
    sentence_predictions: list[dict[str, Any]] | None = None
    long_text_used: bool = False
    long_text_reason: str | None = None
    uncertainty_margin: float | None = None
    model_version: str | None = None
    sentiment_is_uncertain: bool = False
    sentiment_uncertainty_reasons: list[str] = None

    def to_dict(self) -> dict[str, Any]:
        analyzed_at = datetime.now(timezone.utc).isoformat()
        cleaned_scores = {key: float(value) for key, value in self.scores.items()}
        aligned_label, aligned_confidence = _align_label_and_confidence(
            label=self.label,
            scores=cleaned_scores,
            fallback_confidence=self.confidence,
        )
        label_in_scores = aligned_label in cleaned_scores
        score_for_label = float(cleaned_scores.get(aligned_label, aligned_confidence))
        confidence_matches = _is_close(aligned_confidence, score_for_label)
        payload = {
            "label": aligned_label,
            "confidence": float(aligned_confidence),
            "scores": cleaned_scores,
            "reason": self.reason,
            "processed_text": self.processed_text,
            "sentiment_label": aligned_label,
            "sentiment_confidence": float(aligned_confidence),
            "sentiment_scores": cleaned_scores,
            "sentiment_reason": self.reason,
            "sentiment_processed_text": self.processed_text,
            "sentiment_analyzed_at": analyzed_at,
            "long_text_used": self.long_text_used,
            "long_text_reason": self.long_text_reason,
            "uncertainty_margin": self.uncertainty_margin,
            "sentiment_is_uncertain": self.sentiment_is_uncertain,
            "sentiment_uncertainty_reasons": self.sentiment_uncertainty_reasons or [],
            "integrity": {
                "label_in_scores": label_in_scores,
                "confidence_matches_label_score": confidence_matches,
            },
        }
        if self.model_version:
            payload["model_version"] = self.model_version
            payload["sentiment_model_version"] = self.model_version
        if self.review_id is not None:
            payload["review_id"] = self.review_id
            payload["id"] = self.review_id
        if self.sentence_predictions is not None:
            payload["sentence_predictions"] = self.sentence_predictions
        return {key: _clean_value(value) for key, value in payload.items()}

class SentimentAnalysisService:
    def __init__(
        self,
        model_dir: Path | str = Path("sentiment_model/model_artifacts_v9"),
        long_text_mode: str = "auto",
        long_text_min_words: int = 14,
        long_text_min_segments: int = 2,
        close_margin_neutral: float = 0.06,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.long_text_mode = long_text_mode
        self.long_text_min_words = long_text_min_words
        self.long_text_min_segments = long_text_min_segments
        self.close_margin_neutral = close_margin_neutral
        self.preprocessor = IndonesianTextPreprocessor()
        self.model, self.vectorizer, self.metadata, self.hybrid_config = load_artifacts(self.model_dir)
        self.label_order_internal = [str(label) for label in self.metadata.get("label_order", LABEL_ORDER)]
        self.label_order_external = [_external_label(label) for label in self.label_order_internal]

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("version", self.metadata.get("best_model", "unknown")))

    def health(self) -> dict[str, Any]:
        return {
            "success": True,
            "data": {
                "status": "ok",
                "service_ready": True,
                "model_version": self.model_version,
                "model_dir": str(self.model_dir.resolve()),
            },
        }

    def stats(self) -> dict[str, Any]:
        base_metrics = self.metadata.get("base_metrics", {})
        hybrid_metrics = self.metadata.get("hybrid_metrics", {})
        return {
            "success": True,
            "data": {
                "model_version": self.model_version,
                "best_model": self.metadata.get("best_model", "unknown"),
                "data_path": self.metadata.get("data_path", "unknown"),
                "label_order": self.label_order_external,
                "rows_after_preprocessing": self.metadata.get("rows_after_preprocessing"),
                "train_rows_before_rebalance": self.metadata.get("train_rows_before_rebalance"),
                "train_rows_after_rebalance": self.metadata.get("train_rows_after_rebalance"),
                "test_rows": self.metadata.get("test_rows"),
                "selected_class_weight": self.metadata.get("selected_class_weight"),
                "metrics": {
                    "base": base_metrics,
                    "hybrid": hybrid_metrics,
                },
                "hybrid_enabled": bool(self.hybrid_config),
            },
        }

    def predict_single(self, text: str, review_id: Any | None = None) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Field 'text' wajib diisi dan harus berupa string non-kosong.")

        result_df = predict_texts(
            self.model,
            self.vectorizer,
            self.preprocessor,
            [text],
            hybrid_config=self.hybrid_config,
            long_text_mode=self.long_text_mode,
            long_text_min_words=self.long_text_min_words,
            long_text_min_segments=self.long_text_min_segments,
            close_margin_neutral=self.close_margin_neutral,
        )

        row = result_df.iloc[0].to_dict()
        prediction = self._build_prediction(row, review_id=review_id, raw_text=text)
        return {
            "success": True,
            "data": prediction.to_dict(),
        }

    def predict_batch(self, reviews: list[dict[str, Any]], include_sentence_predictions: bool = False) -> dict[str, Any]:
        if not isinstance(reviews, list) or not reviews:
            raise ValueError("Field 'reviews' wajib berupa array yang tidak kosong.")

        normalized_reviews: list[dict[str, Any]] = []
        texts: list[str] = []
        ids: list[Any | None] = []

        for index, item in enumerate(reviews):
            if not isinstance(item, dict):
                raise ValueError(f"reviews[{index}] harus berupa object/dict.")

            review_id = item.get("id", item.get("review_id", index))
            text = item.get("text", item.get("review_text", ""))
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"reviews[{index}].text wajib diisi dan harus string non-kosong.")

            normalized_reviews.append({"id": review_id, "text": text})
            ids.append(review_id)
            texts.append(text)

        result_df = predict_texts(
            self.model,
            self.vectorizer,
            self.preprocessor,
            texts,
            hybrid_config=self.hybrid_config,
            long_text_mode=self.long_text_mode,
            long_text_min_words=self.long_text_min_words,
            long_text_min_segments=self.long_text_min_segments,
            close_margin_neutral=self.close_margin_neutral,
        )

        rows: list[dict[str, Any]] = []
        for idx, (_, result_row) in enumerate(result_df.iterrows()):
            payload = self._build_prediction(
                result_row.to_dict(),
                review_id=ids[idx],
                raw_text=normalized_reviews[idx]["text"],
                include_sentence_predictions=include_sentence_predictions,
            )
            rows.append(payload.to_dict())

        summary = self._summarize_batch(rows)
        return {
            "success": True,
            "data": {
                "reviews": rows,
                "results": rows,
                "count": len(rows),
                "analyzed_count": len(rows),
                "summary": summary,
                "model_version": self.model_version,
            },
        }

    def _build_prediction(
        self,
        row: dict[str, Any],
        review_id: Any | None,
        raw_text: str,
        include_sentence_predictions: bool = True,
    ) -> SentimentPrediction:
        processed_text = str(row.get("ulasan_preprocessed", "") or "")
        label_internal = str(row.get("prediksi_sentimen", "") or "")
        label_external = _external_label(label_internal)
        scores = _scores_from_row(pd.Series(row), self.label_order_internal)
        label_external, confidence = _align_label_and_confidence(label_external, scores)
        reason = _first_non_empty_text(row.get("hybrid_reason"), row.get("long_text_reason"), row.get("reason"))

        long_text_used = bool(row.get("long_text_used", False))
        long_text_reason = row.get("long_text_reason")
        uncertainty_margin = row.get("uncertainty_margin")
        sentiment_is_uncertain = bool(row.get("sentiment_is_uncertain", False))
        sentiment_uncertainty_reasons = row.get("sentiment_uncertainty_reasons", [])

        sentence_predictions: list[dict[str, Any]] | None = None
        if long_text_used and include_sentence_predictions:
            sentence_predictions = self._build_sentence_predictions(raw_text)

        return SentimentPrediction(
            label=label_external,
            confidence=confidence,
            scores=scores,
            reason=reason,
            processed_text=processed_text,
            review_id=review_id,
            sentence_predictions=sentence_predictions,
            long_text_used=long_text_used,
            long_text_reason=str(long_text_reason) if long_text_reason is not None else None,
            uncertainty_margin=float(uncertainty_margin) if uncertainty_margin is not None and not pd.isna(uncertainty_margin) else None,
            model_version=self.model_version,
            sentiment_is_uncertain=sentiment_is_uncertain,
            sentiment_uncertainty_reasons=sentiment_uncertainty_reasons,
        )

    def _build_sentence_predictions(self, text: str) -> list[dict[str, Any]]:
        long_out = classify_long_text(
            model=self.model,
            vectorizer=self.vectorizer,
            preprocessor=self.preprocessor,
            text=text,
            hybrid_config=self.hybrid_config,
        )

        seg_df = long_out["segment_predictions"]
        sentence_predictions: list[dict[str, Any]] = []

        for _, seg in seg_df.iterrows():
            seg_scores = _scores_from_row(seg, self.label_order_internal)
            seg_label_internal = str(seg.get("prediksi_sentimen", "") or "")
            seg_label_external, seg_confidence = _align_label_and_confidence(_external_label(seg_label_internal), seg_scores)
            sentence_predictions.append(
                {
                    "text": seg.get("ulasan_asli", ""),
                    "processed_text": seg.get("ulasan_preprocessed", ""),
                    "label": seg_label_external,
                    "confidence": seg_confidence,
                    "scores": seg_scores,
                    "reason": seg.get("hybrid_reason", "model_only"),
                }
            )

        return sentence_predictions

    @staticmethod
    def _summarize_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
        counts: dict[str, int] = {"negative": 0, "neutral": 0, "positive": 0}
        integrity = {
            "checked": 0,
            "label_in_scores_ok": 0,
            "confidence_matches_label_score_ok": 0,
            "fully_consistent": 0,
        }
        for row in rows:
            label = str(row.get("label", "")).lower()
            if label in counts:
                counts[label] += 1
            integrity_payload = row.get("integrity")
            if isinstance(integrity_payload, dict):
                integrity["checked"] += 1
                label_ok = bool(integrity_payload.get("label_in_scores", False))
                confidence_ok = bool(integrity_payload.get("confidence_matches_label_score", False))
                if label_ok:
                    integrity["label_in_scores_ok"] += 1
                if confidence_ok:
                    integrity["confidence_matches_label_score_ok"] += 1
                if label_ok and confidence_ok:
                    integrity["fully_consistent"] += 1
        return {
            "label_counts": counts,
            "integrity": integrity,
        }
