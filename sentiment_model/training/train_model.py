from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

try:
    from sentiment_model.preprocessing import (
        IndonesianTextPreprocessor,
        LABEL_ORDER,
        prepare_labeled_dataframe,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sentiment_model.preprocessing import (
        IndonesianTextPreprocessor,
        LABEL_ORDER,
        prepare_labeled_dataframe,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training model klasifikasi sentimen dan simpan artefaknya"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("Merged_Excel/dataset_labeled.xlsx"),
        help="Path file dataset berlabel (.xlsx)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sentiment_model/model_artifacts"),
        help="Folder output untuk menyimpan model, vectorizer, dan metadata",
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


def train_and_save(
    data_path: Path,
    output_dir: Path,
    test_size: float,
    random_state: int,
) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path.resolve()}")

    raw_df = pd.read_excel(data_path)
    df = prepare_labeled_dataframe(raw_df, LABEL_ORDER)

    preprocessor = IndonesianTextPreprocessor()
    df["ulasan_preprocessed"] = df["ulasan"].apply(preprocessor.preprocess_text)
    df = df[df["ulasan_preprocessed"].str.strip() != ""].reset_index(drop=True)

    if df.empty:
        raise ValueError("Dataset kosong setelah preprocessing.")

    X = df["ulasan_preprocessed"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "naive_bayes": MultinomialNB(alpha=0.75),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "logistic_regression_balanced": LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
        ),
    }

    metrics: dict[str, dict] = {}

    print("=" * 70)
    print("TRAINING MODEL SENTIMEN")
    print("=" * 70)
    print(f"Dataset             : {data_path.resolve()}")
    print(f"Jumlah data         : {len(df)}")
    print(f"Train/Test          : {len(X_train)} / {len(X_test)}")
    print(f"Label distribution  : {y.value_counts().to_dict()}")
    print("=" * 70)

    for model_name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            labels=LABEL_ORDER,
            digits=4,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)

        macro_f1 = float(report["macro avg"]["f1-score"])
        negative_recall = float(report["negatif"]["recall"])

        metrics[model_name] = {
            "accuracy": float(accuracy),
            "macro_f1": macro_f1,
            "negative_recall": negative_recall,
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
        }

        print(f"\nModel               : {model_name}")
        print(f"Accuracy            : {accuracy:.4f}")
        print(f"Macro F1            : {macro_f1:.4f}")
        print(f"Recall negatif      : {negative_recall:.4f}")

    best_model_name = max(
        metrics,
        key=lambda name: (
            metrics[name]["macro_f1"],
            metrics[name]["negative_recall"],
            metrics[name]["accuracy"],
        ),
    )
    best_model = clone(models[best_model_name])

    # Refit vectorizer dan model terbaik ke seluruh data berlabel agar artefak
    # final memanfaatkan semua sampel terbaru, bukan hanya split training.
    X_all_tfidf = vectorizer.fit_transform(X)
    best_model.fit(X_all_tfidf, y)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "sentiment_model.joblib"
    vectorizer_path = output_dir / "tfidf_vectorizer.joblib"
    metadata_path = output_dir / "metadata.json"

    dump(best_model, model_path)
    dump(vectorizer, vectorizer_path)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(data_path.resolve()),
        "model_path": str(model_path.resolve()),
        "vectorizer_path": str(vectorizer_path.resolve()),
        "best_model": best_model_name,
        "selection_metric": "macro_f1_then_negative_recall_then_accuracy",
        "refit_on_full_data": True,
        "label_order": LABEL_ORDER,
        "rows_after_preprocessing": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": metrics,
    }

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print(f"Model terbaik       : {best_model_name}")
    print(f"Akurasi terbaik     : {metrics[best_model_name]['accuracy']:.4f}")
    print(f"Simpan model        : {model_path.resolve()}")
    print(f"Simpan vectorizer   : {vectorizer_path.resolve()}")
    print(f"Simpan metadata     : {metadata_path.resolve()}")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    train_and_save(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

