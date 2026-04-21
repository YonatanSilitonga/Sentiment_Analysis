from __future__ import annotations

import re
from typing import Any

import pandas as pd

try:
    from sentiment_model.hybrid import apply_hybrid_rules
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sentiment_model.hybrid import apply_hybrid_rules

CONTRAST_SPLIT_PATTERN = re.compile(
    r"(?i)\b(?:"
    r"tapi|tetapi|namun|akan\s+tetapi|"
    r"walaupun|walau|meskipun|meski|biarpun|kendati(?:pun)?|"
    r"hanya\s+saja|cuma(?:n)?|"
    r"padahal|sedangkan|sebaliknya|di\s+sisi\s+lain|sementara\s+itu"
    r")\b"
)

CONFLICT_NEUTRAL_THRESHOLD = 0.30


def split_long_text_segments(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []

    chunks: list[str] = []
    for piece in re.split(r"(?<=[.!?;])\s+", text):
        piece = piece.strip()
        if not piece:
            continue

        # Split further on contrast connectors to isolate mixed sentiment clauses.
        subparts = CONTRAST_SPLIT_PATTERN.split(piece)
        for part in subparts:
            part = part.strip(" ,")
            if part:
                chunks.append(part)

    if not chunks:
        return [text]
    return chunks


def _word_count(text: str) -> int:
    return len([token for token in str(text).strip().split() if token])


def _has_mixed_polarity(segment_predictions: pd.Series) -> bool:
    labels = {str(label) for label in segment_predictions.tolist()}
    return "positif" in labels and "negatif" in labels


def classify_long_text(
    *,
    model,
    vectorizer,
    preprocessor,
    text: str,
    hybrid_config: dict[str, Any] | None,
    uncertainty_margin: float = 0.06,
) -> dict[str, Any]:
    segments = split_long_text_segments(text)
    if not segments:
        segments = [str(text)]

    processed_segments = [preprocessor.preprocess_text(seg) for seg in segments]
    vectors = vectorizer.transform(processed_segments)

    seg_df = pd.DataFrame(
        {
            "ulasan_asli": segments,
            "ulasan_preprocessed": processed_segments,
            "prediksi_sentimen": model.predict(vectors),
        }
    )

    classes = [str(c) for c in model.classes_]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vectors)
        for idx, label in enumerate(classes):
            seg_df[f"prob_{label}"] = probs[:, idx]

    if hybrid_config:
        seg_df = apply_hybrid_rules(seg_df, segments, hybrid_config)

    score_by_label: dict[str, float] = {label: 0.0 for label in classes}
    segment_votes: dict[str, int] = {label: 0 for label in classes}

    for _, row in seg_df.iterrows():
        seg_text = str(row["ulasan_asli"])
        label = str(row["prediksi_sentimen"])
        n_words = max(_word_count(seg_text), 1)

        weight = 1.0 + min(n_words, 40) / 30.0
        if CONTRAST_SPLIT_PATTERN.search(seg_text):
            weight += 0.15

        if "negative_keyword_score" in seg_df.columns:
            weight += min(float(row.get("negative_keyword_score", 0.0)) * 0.04, 0.2)
        if "positive_keyword_score" in seg_df.columns:
            weight += min(float(row.get("positive_keyword_score", 0.0)) * 0.03, 0.15)

        score_by_label[label] = score_by_label.get(label, 0.0) + weight
        segment_votes[label] = segment_votes.get(label, 0) + 1

        for cls in classes:
            prob_col = f"prob_{cls}"
            if prob_col in seg_df.columns:
                score_by_label[cls] = score_by_label.get(cls, 0.0) + float(row[prob_col]) * weight * 0.35

    ranked = sorted(score_by_label.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    final_label = best_label
    reason = "long_text_weighted_aggregate"
    margin = best_score - second_score

    has_contrast_connector = bool(CONTRAST_SPLIT_PATTERN.search(str(text)))
    has_mixed_polarity = _has_mixed_polarity(seg_df["prediksi_sentimen"])

    if has_mixed_polarity and has_contrast_connector and margin <= CONFLICT_NEUTRAL_THRESHOLD:
        final_label = "netral"
        reason = "mixed_polarity_conflict_to_neutral"

    if "netral" in score_by_label and margin <= uncertainty_margin:
        final_label = "netral"
        reason = "long_text_close_scores_to_neutral"

    total_score = sum(score_by_label.values()) or 1.0
    aggregated_probs = {f"prob_{cls}": score_by_label[cls] / total_score for cls in classes}

    return {
        "final_label": final_label,
        "reason": reason,
        "segments": segments,
        "segment_count": len(segments),
        "segment_predictions": seg_df,
        "score_by_label": score_by_label,
        "score_margin": margin,
        "aggregated_probs": aggregated_probs,
        "segment_votes": segment_votes,
    }
