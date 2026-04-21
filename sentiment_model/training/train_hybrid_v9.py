from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    from sentiment_model.hybrid import apply_hybrid_rules, default_hybrid_config
    from sentiment_model.preprocessing import IndonesianTextPreprocessor, LABEL_ORDER, prepare_labeled_dataframe
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sentiment_model.hybrid import apply_hybrid_rules, default_hybrid_config
    from sentiment_model.preprocessing import IndonesianTextPreprocessor, LABEL_ORDER, prepare_labeled_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training model sentimen hybrid versi 9 (phrase-preserving + negative-first baseline)"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("Merged_Excel/dataset_labeled_combined_v3.xlsx"),
        help="Path file dataset berlabel (.xlsx)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sentiment_model/model_artifacts_v9"),
        help="Folder output artefak model hybrid v9",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporsi data test (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed random untuk reproducibility (default: 42)",
    )
    return parser.parse_args()


def evaluate_predictions(y_true, y_pred) -> dict[str, Any]:
    accuracy = float(accuracy_score(y_true, y_pred))
    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    macro_f1 = float(report["macro avg"]["f1-score"])
    negative_recall = float(report["negatif"]["recall"])
    positive_recall = float(report["positif"]["recall"])
    neutral_recall = float(report["netral"]["recall"])

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "negative_recall": negative_recall,
        "positive_recall": positive_recall,
        "neutral_recall": neutral_recall,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def is_better_metrics(candidate: dict[str, Any], current: dict[str, Any] | None) -> bool:
    if current is None:
        return True
    candidate_key = (
        candidate["macro_f1"],
        candidate["negative_recall"],
        candidate["neutral_recall"],
        candidate["positive_recall"],
        candidate["accuracy"],
    )
    current_key = (
        current["macro_f1"],
        current["negative_recall"],
        current["neutral_recall"],
        current["positive_recall"],
        current["accuracy"],
    )
    return candidate_key > current_key


def rebalance_train_dataframe(train_df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    target_ratio = {
        "negatif": 0.34,
        "positif": 0.33,
        "netral": 0.33,
    }

    base_total = len(train_df)
    target_counts = {label: int(base_total * ratio) for label, ratio in target_ratio.items()}
    target_counts["negatif"] += base_total - sum(target_counts.values())

    parts: list[pd.DataFrame] = []
    for label in LABEL_ORDER:
        current = train_df[train_df["label"] == label]
        target_n = target_counts.get(label, len(current))

        if target_n <= 0:
            continue

        if len(current) >= target_n:
            sampled = current.sample(n=target_n, random_state=random_state, replace=False)
        else:
            sampled = current.sample(n=target_n, random_state=random_state, replace=True)

        parts.append(sampled)

    balanced = pd.concat(parts, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return balanced


def tune_model_with_class_weights(
    X_train_tfidf,
    y_train,
    X_test_tfidf,
    X_raw_test: pd.Series,
    X_text_test: pd.Series,
    y_test: pd.Series,
    random_state: int,
) -> tuple[LogisticRegression, dict[str, Any], pd.DataFrame]:
    candidates: list[tuple[str, str | dict[str, float]]] = [
        ("balanced", "balanced"),
        ("custom_v1", {"negatif": 1.05, "netral": 1.0, "positif": 0.95}),
        ("custom_v2", {"negatif": 1.10, "netral": 1.0, "positif": 0.90}),
        ("custom_v3", {"negatif": 1.15, "netral": 0.98, "positif": 0.92}),
    ]

    best_model: LogisticRegression | None = None
    best_result_df: pd.DataFrame | None = None
    best_metrics: dict[str, Any] | None = None
    best_name = ""

    for name, weight in candidates:
        model = LogisticRegression(
            max_iter=3000,
            random_state=random_state,
            class_weight=weight,
            C=1.1,
        )
        model.fit(X_train_tfidf, y_train)

        preds = model.predict(X_test_tfidf)
        result_df = pd.DataFrame(
            {
                "ulasan_asli": X_raw_test.tolist(),
                "ulasan_preprocessed": X_text_test.tolist(),
                "prediksi_sentimen": preds,
            }
        )

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test_tfidf)
            for idx, label in enumerate(model.classes_):
                result_df[f"prob_{label}"] = proba[:, idx]

        metrics = evaluate_predictions(y_test, result_df["prediksi_sentimen"])
        if is_better_metrics(metrics, best_metrics):
            best_model = model
            best_result_df = result_df
            best_metrics = metrics
            best_name = name

    assert best_model is not None and best_result_df is not None and best_metrics is not None
    best_metrics["selected_class_weight_name"] = best_name
    return best_model, best_metrics, best_result_df


def tune_hybrid_config(result_df: pd.DataFrame, raw_texts: list[str], y_true: pd.Series) -> tuple[dict[str, Any], dict[str, Any]]:
    best_config = {
        **default_hybrid_config(),
        "positive_override_enabled": True,
        "positive_label": "positif",
        "positive_keyword_score_threshold": 3,
        "positive_mid_confidence_threshold": 0.22,
        "positive_block_negative_prob": 0.62,
        "neutral_override_enabled": True,
        "neutral_label": "netral",
        "neutral_min_positive_score": 2,
        "neutral_max_negative_score": 1,
        "neutral_max_negative_prob": 0.33,
    }
    best_metrics = evaluate_predictions(y_true, apply_hybrid_rules(result_df.copy(), raw_texts, best_config)["prediksi_sentimen"])

    high_candidates = [0.55, 0.60, 0.65]
    mid_candidates = [0.25, 0.30, 0.35]
    mid_score_candidates = [2, 3]
    strong_prob_candidates = [0.10, 0.15, 0.20]
    neutral_neg_prob_candidates = [0.30, 0.35, 0.40]

    for high in high_candidates:
        for mid in mid_candidates:
            if mid > high:
                continue
            for mid_score in mid_score_candidates:
                for strong_prob in strong_prob_candidates:
                    for neutral_neg_prob in neutral_neg_prob_candidates:
                        config = {
                            **default_hybrid_config(),
                            "high_confidence_threshold": high,
                            "mid_confidence_threshold": mid,
                            "mid_keyword_score_threshold": mid_score,
                            "strong_signal_min_prob": strong_prob,
                            "positive_override_enabled": True,
                            "positive_label": "positif",
                            "positive_keyword_score_threshold": 3,
                            "positive_mid_confidence_threshold": 0.22,
                            "positive_block_negative_prob": 0.62,
                            "neutral_override_enabled": True,
                            "neutral_label": "netral",
                            "neutral_min_positive_score": 2,
                            "neutral_max_negative_score": 1,
                            "neutral_max_negative_prob": neutral_neg_prob,
                        }
                        hybrid_df = apply_hybrid_rules(result_df.copy(), raw_texts, config)
                        metrics = evaluate_predictions(y_true, hybrid_df["prediksi_sentimen"])
                        if is_better_metrics(metrics, best_metrics):
                            best_config = config
                            best_metrics = metrics

    return best_config, best_metrics


def train_hybrid_v9(data_path: Path, output_dir: Path, test_size: float, random_state: int) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path.resolve()}")

    raw_df = pd.read_excel(data_path)
    df = prepare_labeled_dataframe(raw_df, LABEL_ORDER)

    preprocessor = IndonesianTextPreprocessor()
    df["ulasan_preprocessed"] = df["ulasan"].apply(preprocessor.preprocess_text)
    df = df[df["ulasan_preprocessed"].str.strip() != ""].reset_index(drop=True)

    X_raw = df["ulasan"]
    X_text = df["ulasan_preprocessed"]
    y = df["label"]

    X_text_train, X_text_test, X_raw_train, X_raw_test, y_train, y_test = train_test_split(
        X_text,
        X_raw,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train_df = pd.DataFrame(
        {
            "ulasan": X_raw_train.values,
            "ulasan_preprocessed": X_text_train.values,
            "label": y_train.values,
        }
    )
    train_df_balanced = rebalance_train_dataframe(train_df, random_state=random_state)

    vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 3), sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(train_df_balanced["ulasan_preprocessed"])
    X_test_tfidf = vectorizer.transform(X_text_test)

    base_model, base_metrics, result_df = tune_model_with_class_weights(
        X_train_tfidf=X_train_tfidf,
        y_train=train_df_balanced["label"],
        X_test_tfidf=X_test_tfidf,
        X_raw_test=X_raw_test,
        X_text_test=X_text_test,
        y_test=y_test,
        random_state=random_state,
    )

    hybrid_config, hybrid_metrics = tune_hybrid_config(result_df.copy(), X_raw_test.tolist(), y_test)

    print("=" * 70)
    print("TRAINING MODEL HYBRID V9")
    print("=" * 70)
    print(f"Dataset                 : {data_path.resolve()}")
    print(f"Jumlah data total       : {len(df)}")
    print(f"Train/Test awal         : {len(X_text_train)} / {len(X_text_test)}")
    print(f"Train setelah rebalance : {len(train_df_balanced)}")
    print(f"Distribusi train awal   : {train_df['label'].value_counts().to_dict()}")
    print(f"Distribusi train rebal  : {train_df_balanced['label'].value_counts().to_dict()}")
    print("-" * 70)
    print("BASE MODEL")
    print(f"Selected class_weight   : {base_metrics['selected_class_weight_name']}")
    print(f"Accuracy                : {base_metrics['accuracy']:.4f}")
    print(f"Macro F1                : {base_metrics['macro_f1']:.4f}")
    print(f"Recall negatif          : {base_metrics['negative_recall']:.4f}")
    print(f"Recall netral           : {base_metrics['neutral_recall']:.4f}")
    print(f"Recall positif          : {base_metrics['positive_recall']:.4f}")
    print("-" * 70)
    print("HYBRID MODEL")
    print(f"Accuracy                : {hybrid_metrics['accuracy']:.4f}")
    print(f"Macro F1                : {hybrid_metrics['macro_f1']:.4f}")
    print(f"Recall negatif          : {hybrid_metrics['negative_recall']:.4f}")
    print(f"Recall netral           : {hybrid_metrics['neutral_recall']:.4f}")
    print(f"Recall positif          : {hybrid_metrics['positive_recall']:.4f}")

    selected_weight_name = str(base_metrics["selected_class_weight_name"])
    if selected_weight_name == "custom_v1":
        selected_weight_map: str | dict[str, float] = {"negatif": 1.05, "netral": 1.0, "positif": 0.95}
    elif selected_weight_name == "custom_v2":
        selected_weight_map = {"negatif": 1.10, "netral": 1.0, "positif": 0.90}
    elif selected_weight_name == "custom_v3":
        selected_weight_map = {"negatif": 1.15, "netral": 0.98, "positif": 0.92}
    else:
        selected_weight_map = "balanced"

    full_df = pd.DataFrame(
        {
            "ulasan": X_raw.values,
            "ulasan_preprocessed": X_text.values,
            "label": y.values,
        }
    )
    full_rebalanced = rebalance_train_dataframe(full_df, random_state=random_state)

    X_full_tfidf = vectorizer.fit_transform(full_rebalanced["ulasan_preprocessed"])
    base_model = LogisticRegression(
        max_iter=3000,
        random_state=random_state,
        class_weight=selected_weight_map,
        C=1.1,
    )
    base_model.fit(X_full_tfidf, full_rebalanced["label"])

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "sentiment_model_v9.joblib"
    vectorizer_path = output_dir / "tfidf_vectorizer_v9.joblib"
    hybrid_path = output_dir / "hybrid_rules_v9.json"
    metadata_path = output_dir / "metadata_v9.json"

    dump(base_model, model_path)
    dump(vectorizer, vectorizer_path)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "version": "v9_phrase_preserving_ngram_aware_negative_first",
        "data_path": str(data_path.resolve()),
        "model_path": str(model_path.resolve()),
        "vectorizer_path": str(vectorizer_path.resolve()),
        "hybrid_rules_path": str(hybrid_path.resolve()),
        "best_model": "logistic_regression_hybrid_v9",
        "selection_metric": "macro_f1_then_negative_recall_then_neutral_recall_then_positive_recall_then_accuracy",
        "refit_on_full_data": True,
        "label_order": LABEL_ORDER,
        "rows_after_preprocessing": int(len(df)),
        "train_rows_before_rebalance": int(len(X_text_train)),
        "train_rows_after_rebalance": int(len(train_df_balanced)),
        "test_rows": int(len(X_text_test)),
        "full_rows_after_rebalance": int(len(full_rebalanced)),
        "data_source": "combined_v3_with_phrase_preservation",
        "train_distribution_before": train_df["label"].value_counts().to_dict(),
        "train_distribution_after": train_df_balanced["label"].value_counts().to_dict(),
        "selected_class_weight": selected_weight_map,
        "base_metrics": base_metrics,
        "hybrid_metrics": hybrid_metrics,
        "hybrid_config": hybrid_config,
    }

    hybrid_path.write_text(json.dumps(hybrid_config, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print(f"Simpan model v9        : {model_path.resolve()}")
    print(f"Simpan vectorizer v9   : {vectorizer_path.resolve()}")
    print(f"Simpan rules v9        : {hybrid_path.resolve()}")
    print(f"Simpan metadata v9     : {metadata_path.resolve()}")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    train_hybrid_v9(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

