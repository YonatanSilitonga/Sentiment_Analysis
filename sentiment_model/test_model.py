from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from joblib import load

try:
    from sentiment_model.hybrid import apply_hybrid_rules
    from sentiment_model.long_text import CONTRAST_SPLIT_PATTERN, classify_long_text, split_long_text_segments
    from sentiment_model.preprocessing import IndonesianTextPreprocessor
    from sentiment_model.utils import TEXT_COLUMN_CANDIDATES, choose_text_column
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sentiment_model.hybrid import apply_hybrid_rules
    from sentiment_model.long_text import CONTRAST_SPLIT_PATTERN, classify_long_text, split_long_text_segments
    from sentiment_model.preprocessing import IndonesianTextPreprocessor
    from sentiment_model.utils import TEXT_COLUMN_CANDIDATES, choose_text_column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Testing model sentimen dari terminal (single, interaktif, atau batch file)"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("sentiment_model/model_artifacts"),
        help="Folder model_artifacts hasil training",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Teks ulasan untuk prediksi single",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="File batch input (.xlsx/.csv/.txt)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default=None,
        help="Nama kolom teks pada input file batch",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="File output hasil prediksi batch (.xlsx/.csv)",
    )
    parser.add_argument(
        "--long-text-mode",
        type=str,
        choices=["off", "auto", "always"],
        default="auto",
        help="Mode inferensi teks panjang: off, auto (hanya teks panjang), always",
    )
    parser.add_argument(
        "--long-text-min-words",
        type=int,
        default=14,
        help="Minimal jumlah kata agar mode auto dianggap teks panjang",
    )
    parser.add_argument(
        "--long-text-min-segments",
        type=int,
        default=2,
        help="Minimal jumlah segmen agar mode auto memakai agregasi teks panjang",
    )
    parser.add_argument(
        "--close-margin-neutral",
        type=float,
        default=0.06,
        help="Jika margin probabilitas top-1 vs top-2 <= nilai ini, pakai label netral (0 untuk nonaktif)",
    )
    return parser.parse_args()


def load_artifacts(model_dir: Path):
    model_candidates = ["sentiment_model_v9.joblib", "sentiment_model.joblib", "sentiment_model_v2.joblib", "sentiment_model_v3.joblib", "sentiment_model_v4.joblib", "sentiment_model_v5.joblib", "sentiment_model_v7.joblib", "sentiment_model_v8.joblib"]
    vectorizer_candidates = ["tfidf_vectorizer_v9.joblib", "tfidf_vectorizer.joblib", "tfidf_vectorizer_v2.joblib", "tfidf_vectorizer_v3.joblib", "tfidf_vectorizer_v4.joblib", "tfidf_vectorizer_v5.joblib", "tfidf_vectorizer_v7.joblib", "tfidf_vectorizer_v8.joblib"]
    metadata_candidates = ["metadata_v9.json", "metadata.json", "metadata_v2.json", "metadata_v3.json", "metadata_v4.json", "metadata_v5.json", "metadata_v7.json", "metadata_v8.json"]
    hybrid_candidates = ["hybrid_rules_v9.json", "hybrid_rules.json", "hybrid_rules_v2.json", "hybrid_rules_v3.json", "hybrid_rules_v4.json", "hybrid_rules_v5.json", "hybrid_rules_v7.json", "hybrid_rules_v8.json"]

    metadata_path = next((model_dir / name for name in metadata_candidates if (model_dir / name).exists()), model_dir / metadata_candidates[0])
    hybrid_path = next((model_dir / name for name in hybrid_candidates if (model_dir / name).exists()), None)

    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    metadata_model_path = Path(str(metadata.get("model_path", ""))) if metadata.get("model_path") else None
    metadata_vectorizer_path = Path(str(metadata.get("vectorizer_path", ""))) if metadata.get("vectorizer_path") else None

    if metadata_model_path is not None and metadata_model_path.exists():
        model_path = metadata_model_path
    else:
        model_path = next((model_dir / name for name in model_candidates if (model_dir / name).exists()), model_dir / model_candidates[0])

    if metadata_vectorizer_path is not None and metadata_vectorizer_path.exists():
        vectorizer_path = metadata_vectorizer_path
    else:
        vectorizer_path = next((model_dir / name for name in vectorizer_candidates if (model_dir / name).exists()), model_dir / vectorizer_candidates[0])

    if not model_path.exists() or not vectorizer_path.exists():
        legacy_dir = Path("model_artifacts")
        legacy_model = legacy_dir / "sentiment_model.joblib"
        legacy_vectorizer = legacy_dir / "tfidf_vectorizer.joblib"
        legacy_metadata = legacy_dir / "metadata.json"
        if legacy_model.exists() and legacy_vectorizer.exists():
            model_path = legacy_model
            vectorizer_path = legacy_vectorizer
            metadata_path = legacy_metadata
            hybrid_path = None
        else:
            raise FileNotFoundError(
                "Artefak model tidak ditemukan. Jalankan training dulu: "
                "python -m sentiment_model.train_model"
            )

    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            "Artefak model tidak ditemukan. Jalankan training dulu: "
            "python -m sentiment_model.train_model"
        )

    model = load(model_path)
    vectorizer = load(vectorizer_path)
    hybrid_config = None

    if not metadata and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if hybrid_path is not None and hybrid_path.exists():
        hybrid_config = json.loads(hybrid_path.read_text(encoding="utf-8"))

    if not hasattr(model, "predict"):
        raise ValueError(f"Model artifact tidak valid atau tidak kompatibel: {model_path}")

    if not hasattr(vectorizer, "transform"):
        raise ValueError(f"Vectorizer artifact tidak valid atau tidak kompatibel: {vectorizer_path}")

    vocab = getattr(vectorizer, "vocabulary_", None)
    if not isinstance(vocab, dict) or len(vocab) == 0:
        raise ValueError(
            "idf vector is not fitted. Artifact TF-IDF tidak ter-fit atau tidak kompatibel. "
            f"Periksa file vectorizer: {vectorizer_path}"
        )

    if hasattr(vectorizer, "idf_"):
        idf_values = getattr(vectorizer, "idf_", None)
        idf_size = getattr(idf_values, "size", None)
        if idf_values is None or idf_size in (None, 0):
            raise ValueError(
                "idf vector is not fitted. Artifact TF-IDF tidak ter-fit atau tidak kompatibel. "
                f"Periksa file vectorizer: {vectorizer_path}"
            )

    return model, vectorizer, metadata, hybrid_config


def predict_texts(
    model,
    vectorizer,
    preprocessor: IndonesianTextPreprocessor,
    texts: list[str],
    hybrid_config: dict | None = None,
    long_text_mode: str = "auto",
    long_text_min_words: int = 18,
    long_text_min_segments: int = 2,
    close_margin_neutral: float = 0.06,
) -> pd.DataFrame:
    rows: list[dict] = []

    for text in texts:
        raw_text = str(text)
        processed_text = preprocessor.preprocess_text(raw_text)
        segs = split_long_text_segments(raw_text)
        word_count = len([t for t in raw_text.split() if t])
        has_contrast_connector = bool(re.search(CONTRAST_SPLIT_PATTERN, raw_text))

        use_long_text = long_text_mode == "always"
        if long_text_mode == "auto":
            use_long_text = (
                (word_count >= long_text_min_words or len(raw_text) >= 100)
                and len(segs) >= long_text_min_segments
            ) or (has_contrast_connector and len(segs) >= 2)

        if use_long_text:
            long_out = classify_long_text(
                model=model,
                vectorizer=vectorizer,
                preprocessor=preprocessor,
                text=raw_text,
                hybrid_config=hybrid_config,
            )
            row = {
                "ulasan_asli": raw_text,
                "ulasan_preprocessed": processed_text,
                "prediksi_sentimen": long_out["final_label"],
                "long_text_used": True,
                "long_text_segments": long_out["segment_count"],
                "long_text_margin": float(long_out["score_margin"]),
                "long_text_reason": long_out["reason"],
                "uncertainty_margin": float(long_out["score_margin"]),
                "uncertainty_policy": "long_text_aggregate",
            }
            row.update(long_out["aggregated_probs"])
            rows.append(row)
            continue

        vectors = vectorizer.transform([processed_text])
        prediction = str(model.predict(vectors)[0])

        single_df = pd.DataFrame(
            {
                "ulasan_asli": [raw_text],
                "ulasan_preprocessed": [processed_text],
                "prediksi_sentimen": [prediction],
            }
        )
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vectors)[0]
            for idx, label in enumerate(model.classes_):
                single_df[f"prob_{label}"] = [proba[idx]]

        if hybrid_config:
            single_df = apply_hybrid_rules(single_df, [raw_text], hybrid_config)

        # Conservative policy: when top probabilities are very close,
        # avoid hard positive/negative label and map to neutral.
        if close_margin_neutral > 0:
            prob_cols = [col for col in single_df.columns if col.startswith("prob_")]
            if len(prob_cols) >= 2:
                scored = sorted(
                    [(col.replace("prob_", ""), float(single_df.at[0, col])) for col in prob_cols],
                    key=lambda x: x[1],
                    reverse=True,
                )
                top_label, top_prob = scored[0]
                second_prob = scored[1][1]
                margin = top_prob - second_prob

                current_reason = str(single_df.at[0, "hybrid_reason"]) if "hybrid_reason" in single_df.columns else "model_only"
                if margin <= close_margin_neutral and current_reason == "model_only":
                    single_df.at[0, "prediksi_sentimen"] = "netral"
                    single_df.at[0, "hybrid_reason"] = "close_probability_to_neutral"
                single_df["uncertainty_margin"] = [margin]
                single_df["uncertainty_policy"] = [f"close_margin<={close_margin_neutral}"]

        row = single_df.iloc[0].to_dict()
        row["long_text_used"] = False
        row["long_text_segments"] = len(segs)
        row["long_text_margin"] = None
        row["long_text_reason"] = "short_text_default"
        if "uncertainty_margin" not in row:
            row["uncertainty_margin"] = None
        if "uncertainty_policy" not in row:
            row["uncertainty_policy"] = "disabled"
        rows.append(row)

    return pd.DataFrame(rows)


def read_batch_input(input_file: Path) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"File batch tidak ditemukan: {input_file.resolve()}")

    suffix = input_file.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(input_file)
    if suffix == ".csv":
        return pd.read_csv(input_file)
    if suffix == ".txt":
        lines = input_file.read_text(encoding="utf-8").splitlines()
        return pd.DataFrame({"ulasan": [line.strip() for line in lines if line.strip()]})

    raise ValueError("Format file batch belum didukung. Gunakan .xlsx, .csv, atau .txt")


def save_batch_output(result_df: pd.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.suffix.lower() == ".csv":
        result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    else:
        result_df.to_excel(output_file, index=False)


def run_single_or_interactive(
    model,
    vectorizer,
    preprocessor: IndonesianTextPreprocessor,
    text: str | None,
    hybrid_config: dict | None,
    long_text_mode: str,
    long_text_min_words: int,
    long_text_min_segments: int,
    close_margin_neutral: float,
) -> None:
    if text:
        result_df = predict_texts(
            model,
            vectorizer,
            preprocessor,
            [text],
            hybrid_config=hybrid_config,
            long_text_mode=long_text_mode,
            long_text_min_words=long_text_min_words,
            long_text_min_segments=long_text_min_segments,
            close_margin_neutral=close_margin_neutral,
        )
        row = result_df.iloc[0]
        print("=" * 70)
        print("HASIL PREDIKSI SINGLE")
        print("=" * 70)
        print(f"Ulasan asli         : {row['ulasan_asli']}")
        print(f"Teks preprocessed   : {row['ulasan_preprocessed']}")
        print(f"Prediksi sentimen   : {row['prediksi_sentimen']}")

        prob_columns = [col for col in result_df.columns if col.startswith("prob_")]
        if prob_columns:
            print("Probabilitas label:")
            for col in prob_columns:
                print(f"- {col.replace('prob_', ''):<10}: {row[col]:.4f}")
        if "prediksi_sentimen_base" in result_df.columns:
            print(f"Prediksi base       : {row['prediksi_sentimen_base']}")
            print(f"Hybrid reason       : {row['hybrid_reason']}")
        print(f"Long-text mode      : {bool(row.get('long_text_used', False))}")
        print(f"Long-text reason    : {row.get('long_text_reason', '-')}")
        if row.get("uncertainty_margin") is not None:
            print(f"Uncertainty margin  : {float(row.get('uncertainty_margin')):.4f}")
        print("=" * 70)
        return

    print("Mode interaktif aktif. Ketik `exit` untuk berhenti.")
    while True:
        user_text = input("Masukkan ulasan > ").strip()
        if user_text.lower() in {"exit", "quit"}:
            print("Selesai.")
            break
        if not user_text:
            print("Ulasan kosong. Coba lagi.")
            continue

        result_df = predict_texts(
            model,
            vectorizer,
            preprocessor,
            [user_text],
            hybrid_config=hybrid_config,
            long_text_mode=long_text_mode,
            long_text_min_words=long_text_min_words,
            long_text_min_segments=long_text_min_segments,
            close_margin_neutral=close_margin_neutral,
        )
        row = result_df.iloc[0]
        print(f"Prediksi            : {row['prediksi_sentimen']}")
        if "prediksi_sentimen_base" in result_df.columns:
            print(f"Prediksi base       : {row['prediksi_sentimen_base']}")
            print(f"Hybrid reason       : {row['hybrid_reason']}")

        prob_columns = [col for col in result_df.columns if col.startswith("prob_")]
        if prob_columns:
            prob_text = ", ".join(
                f"{col.replace('prob_', '')}={row[col]:.4f}"
                for col in prob_columns
            )
            print(f"Probabilitas        : {prob_text}")
        print(f"Long-text mode      : {bool(row.get('long_text_used', False))} ({row.get('long_text_reason', '-')})")
        if row.get("uncertainty_margin") is not None:
            print(f"Uncertainty margin  : {float(row.get('uncertainty_margin')):.4f}")


def run_batch(
    model,
    vectorizer,
    preprocessor: IndonesianTextPreprocessor,
    input_file: Path,
    text_column: str | None,
    output_file: Path | None,
    hybrid_config: dict | None,
    long_text_mode: str,
    long_text_min_words: int,
    long_text_min_segments: int,
    close_margin_neutral: float,
) -> None:
    batch_df = read_batch_input(input_file)
    selected_column = choose_text_column(
        batch_df,
        explicit_column=text_column,
        candidates=TEXT_COLUMN_CANDIDATES,
    )

    texts = batch_df[selected_column].fillna("").astype(str).tolist()
    result_df = predict_texts(
        model,
        vectorizer,
        preprocessor,
        texts,
        hybrid_config=hybrid_config,
        long_text_mode=long_text_mode,
        long_text_min_words=long_text_min_words,
        long_text_min_segments=long_text_min_segments,
        close_margin_neutral=close_margin_neutral,
    )

    merged = batch_df.copy()
    merged["ulasan_preprocessed"] = result_df["ulasan_preprocessed"]
    merged["prediksi_sentimen"] = result_df["prediksi_sentimen"]

    for column in result_df.columns:
        if column.startswith("prob_"):
            merged[column] = result_df[column]

    for column in ["long_text_used", "long_text_segments", "long_text_margin", "long_text_reason"]:
        if column in result_df.columns:
            merged[column] = result_df[column]

    for column in ["uncertainty_margin", "uncertainty_policy"]:
        if column in result_df.columns:
            merged[column] = result_df[column]

    if output_file is None:
        output_file = input_file.with_name(f"{input_file.stem}_predictions.xlsx")

    save_batch_output(merged, output_file)

    print("=" * 70)
    print("HASIL PREDIKSI BATCH")
    print("=" * 70)
    print(f"Input file          : {input_file.resolve()}")
    print(f"Kolom teks          : {selected_column}")
    print(f"Jumlah baris        : {len(merged)}")
    print(f"Output file         : {output_file.resolve()}")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    model, vectorizer, metadata, hybrid_config = load_artifacts(args.model_dir)
    preprocessor = IndonesianTextPreprocessor()

    if metadata:
        print(f"Model terbaik       : {metadata.get('best_model', 'unknown')}")
        print(f"Dilatih dari data   : {metadata.get('data_path', 'unknown')}")
    if hybrid_config:
        print("Hybrid inference    : aktif")

    if args.input_file is not None:
        run_batch(
            model=model,
            vectorizer=vectorizer,
            preprocessor=preprocessor,
            input_file=args.input_file,
            text_column=args.text_column,
            output_file=args.output_file,
            hybrid_config=hybrid_config,
            long_text_mode=args.long_text_mode,
            long_text_min_words=args.long_text_min_words,
            long_text_min_segments=args.long_text_min_segments,
            close_margin_neutral=args.close_margin_neutral,
        )
        return

    run_single_or_interactive(
        model=model,
        vectorizer=vectorizer,
        preprocessor=preprocessor,
        text=args.text,
        hybrid_config=hybrid_config,
        long_text_mode=args.long_text_mode,
        long_text_min_words=args.long_text_min_words,
        long_text_min_segments=args.long_text_min_segments,
        close_margin_neutral=args.close_margin_neutral,
    )


if __name__ == "__main__":
    main()
