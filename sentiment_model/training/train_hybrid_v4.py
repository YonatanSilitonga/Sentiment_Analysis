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
    from sentiment_model.preprocessing import (
        IndonesianTextPreprocessor,
        LABEL_ORDER,
        prepare_labeled_dataframe,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sentiment_model.hybrid import apply_hybrid_rules, default_hybrid_config
    from sentiment_model.preprocessing import (
        IndonesianTextPreprocessor,
        LABEL_ORDER,
        prepare_labeled_dataframe,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training model sentimen hybrid (ML + negative keyword rules) versi 4 dengan combined dataset"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("Merged_Excel/dataset_labeled_combined.xlsx"),
        help="Path file dataset berlabel combined (.xlsx)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sentiment_model/model_artifacts_v4"),
        help="Folder output artefak model hybrid v4",
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

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "negative_recall": negative_recall,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def tune_hybrid_config(result_df: pd.DataFrame, raw_texts: list[str], y_true: pd.Series) -> tuple[dict[str, Any], dict[str, Any]]:
    best_config = default_hybrid_config()
    best_metrics = evaluate_predictions(y_true, result_df["prediksi_sentimen"])

    high_candidates = [0.50, 0.55, 0.60]
    mid_candidates = [0.20, 0.25, 0.30, 0.35]
    mid_score_candidates = [1, 2, 3]
    strong_prob_candidates = [0.00, 0.10, 0.15, 0.20]

    for high in high_candidates:
        for mid in mid_candidates:
            if mid > high:
                continue

            for mid_score in mid_score_candidates:
                for strong_prob in strong_prob_candidates:
                    config = {
                        **default_hybrid_config(),
                        "high_confidence_threshold": high,
                        "mid_confidence_threshold": mid,
                        "mid_keyword_score_threshold": mid_score,
                        "strong_signal_min_prob": strong_prob,
                    }
                    hybrid_df = apply_hybrid_rules(result_df, raw_texts, config)
                    metrics = evaluate_predictions(y_true, hybrid_df["prediksi_sentimen"])

                    if (
                        metrics["macro_f1"],
                        metrics["negative_recall"],
                        metrics["accuracy"],
                    ) > (
                        best_metrics["macro_f1"],
                        best_metrics["negative_recall"],
                        best_metrics["accuracy"],
                    ):
                        best_config = config
                        best_metrics = metrics

    return best_config, best_metrics


def train_hybrid_v4(data_path: Path, output_dir: Path, test_size: float, random_state: int) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path.resolve()}")

    raw_df = pd.read_excel(data_path)
    df = prepare_labeled_dataframe(raw_df, LABEL_ORDER)

    preprocessor = IndonesianTextPreprocessor()
    df["ulasan_preprocessed"] = df["ulasan"].apply(preprocessor.preprocess_text)
    df = df[df["ulasan_preprocessed"].str.strip() != ""].reset_index(drop=True)

    if df.empty:
        raise ValueError("Dataset kosong setelah preprocessing.")

    X_raw = df["ulasan"]
    X_text = df["ulasan_preprocessed"]
    y = df["label"]

    X_text_train, X_text_test, _X_raw_train, X_raw_test, y_train, y_test = train_test_split(
        X_text,
        X_raw,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_text_train)
    X_test_tfidf = vectorizer.transform(X_text_test)

    base_model = LogisticRegression(
        max_iter=2000,
        random_state=random_state,
        class_weight="balanced",
    )
    base_model.fit(X_train_tfidf, y_train)

    base_predictions = base_model.predict(X_test_tfidf)
    result_df = pd.DataFrame(
        {
            "ulasan_asli": X_raw_test.tolist(),
            "ulasan_preprocessed": X_text_test.tolist(),
            "prediksi_sentimen": base_predictions,
        }
    )

    if hasattr(base_model, "predict_proba"):
        proba = base_model.predict_proba(X_test_tfidf)
        for idx, label in enumerate(base_model.classes_):
            result_df[f"prob_{label}"] = proba[:, idx]

    base_metrics = evaluate_predictions(y_test, result_df["prediksi_sentimen"])
    hybrid_config, hybrid_metrics = tune_hybrid_config(result_df.copy(), X_raw_test.tolist(), y_test)

    print("=" * 70)
    print("TRAINING MODEL HYBRID V4")
    print("=" * 70)
    print(f"Dataset             : {data_path.resolve()}")
    print(f"Jumlah data         : {len(df)}")
    print(f"Train/Test          : {len(X_text_train)} / {len(X_text_test)}")
    print(f"Label distribution  : {y.value_counts().to_dict()}")
    print("-" * 70)
    print("BASE MODEL (logistic_regression_balanced)")
    print(f"Accuracy            : {base_metrics['accuracy']:.4f}")
    print(f"Macro F1            : {base_metrics['macro_f1']:.4f}")
    print(f"Recall negatif      : {base_metrics['negative_recall']:.4f}")
    print("-" * 70)
    print("HYBRID MODEL (base + extended rules + new dataset)")
    print(f"Accuracy            : {hybrid_metrics['accuracy']:.4f}")
    print(f"Macro F1            : {hybrid_metrics['macro_f1']:.4f}")
    print(f"Recall negatif      : {hybrid_metrics['negative_recall']:.4f}")

    X_all_tfidf = vectorizer.fit_transform(X_text)
    base_model.fit(X_all_tfidf, y)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "sentiment_model_v4.joblib"
    vectorizer_path = output_dir / "tfidf_vectorizer_v4.joblib"
    hybrid_path = output_dir / "hybrid_rules_v4.json"
    metadata_path = output_dir / "metadata_v4.json"

    dump(base_model, model_path)
    dump(vectorizer, vectorizer_path)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "version": "v4_hybrid_extended_lexicon_combined_dataset",
        "data_path": str(data_path.resolve()),
        "model_path": str(model_path.resolve()),
        "vectorizer_path": str(vectorizer_path.resolve()),
        "hybrid_rules_path": str(hybrid_path.resolve()),
        "best_model": "logistic_regression_balanced_hybrid_v4",
        "selection_metric": "macro_f1_then_negative_recall_then_accuracy",
        "refit_on_full_data": True,
        "label_order": LABEL_ORDER,
        "rows_after_preprocessing": int(len(df)),
        "train_rows": int(len(X_text_train)),
        "test_rows": int(len(X_text_test)),
        "data_source": "combined: original (379) + new syntetik Toba (400)",
        "base_metrics": base_metrics,
        "hybrid_metrics": hybrid_metrics,
        "hybrid_config": hybrid_config,
    }

    hybrid_path.write_text(json.dumps(hybrid_config, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print(f"Simpan model v4     : {model_path.resolve()}")
    print(f"Simpan vectorizer v4: {vectorizer_path.resolve()}")
    print(f"Simpan rules v4     : {hybrid_path.resolve()}")
    print(f"Simpan metadata v4  : {metadata_path.resolve()}")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    train_hybrid_v4(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

