from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

try:
    from sentiment_model.preprocessing import IndonesianTextPreprocessor, strip_edge_ellipsis
    from sentiment_model.test_model import load_artifacts, predict_texts
    from sentiment_model.utils import (
        TEXT_COLUMN_CANDIDATES,
        choose_text_column,
        find_column,
        normalize_spaces,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sentiment_model.preprocessing import IndonesianTextPreprocessor, strip_edge_ellipsis
    from sentiment_model.test_model import load_artifacts, predict_texts
    from sentiment_model.utils import (
        TEXT_COLUMN_CANDIDATES,
        choose_text_column,
        find_column,
        normalize_spaces,
    )

NAME_COLUMN_CANDIDATES = ["nama", "title", "name"]
DATE_COLUMN_CANDIDATES = ["tanggal", "data", "date"]

NEGATIVE_SIGNALS = [
    (r"\bpenipu\b|\btipu(?:an)?\b|\bpungli\b", 3, "penipu_or_tipuan"),
    (r"\b(?:gak|ga|nggak|tidak|tak)\s+(?:rekomen|rekom|rekomendasi|recommended)\b", 3, "not_recommended"),
    (r"\b(?:tidak|gak|ga|nggak)\s+ramah\b", 3, "not_friendly"),
    (r"\b(?:tidak|gak|ga|nggak)\s+sopan\b", 3, "not_polite"),
    (r"\b(?:tidak|gak|ga|nggak)\s+aman\b|\bgak amanah\b|\btidak amanah\b", 3, "unsafe_or_untrustworthy"),
    (r"\b(?:tidak|gak|ga|nggak|tak)\s+boleh\b", 3, "not_allowed"),
    (r"\b(?:dipulang(?:kan)?|disuruh pulang|ditolak)\b", 3, "turned_away"),
    (r"\btidak memuaskan\b|\bkurang memuaskan\b", 3, "unsatisfactory"),
    (r"\b(?:gak|ga|nggak|tidak)\s+akan\s+kesini\s+lagi\b", 3, "wont_return"),
    (r"\bmaling\b|\bcuri\b", 3, "theft_risk"),
    (r"\bparah\b", 2, "bad_experience"),
    (r"\bsombong\b|\bsewot\b|\bjutek\b", 2, "bad_attitude"),
    (r"\bmahal\b|\bgak masuk akal\b|\btidak masuk akal\b", 2, "overpriced"),
    (r"\bgembok\b|\bdigembok\b", 2, "locked_facility"),
    (r"\bkotor\b|\bjijik\b", 2, "dirty_place"),
    (r"\bilegal\b", 2, "illegal_fee"),
    (r"\brusak\b|\bbahaya\b|\bcuram\b", 1, "bad_access"),
]

METADATA_COLUMNS = {
    "web_scraper_order",
    "web_scraper_start_url",
    "image",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tambahkan kandidat ulasan baru ke dataset_labeled dengan filter negatif konservatif"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("Merged_Excel/dataset_labeled.xlsx"),
        help="Path dataset berlabel yang akan diperbarui",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("Merged_Excel"),
        help="Folder yang berisi file Ulasan *.xlsx",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="Ulasan *.xlsx",
        help="Glob pattern file sumber ulasan baru",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("sentiment_model/model_artifacts"),
        help="Folder artefak model lama untuk bantu scoring kandidat",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("sentiment_model/model_artifacts/augmentation_report.xlsx"),
        help="File Excel report kandidat augmentasi",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Hanya tampilkan kandidat dan report tanpa menyimpan ke dataset_labeled",
    )
    return parser.parse_args()


def make_review_key(text: str) -> str:
    lowered = strip_edge_ellipsis(str(text)).lower()
    lowered = normalize_spaces(lowered)
    return lowered


def extract_candidates(source_dir: Path, pattern: str, existing_keys: set[str]) -> pd.DataFrame:
    records: list[dict[str, str]] = []

    for file_path in sorted(source_dir.glob(pattern)):
        df = pd.read_excel(file_path)
        if df.empty:
            continue

        text_column = choose_text_column(
            df,
            candidates=TEXT_COLUMN_CANDIDATES,
            metadata_columns=METADATA_COLUMNS,
            use_scoring_fallback=True,
        )
        name_column = find_column(df, NAME_COLUMN_CANDIDATES)
        date_column = find_column(df, DATE_COLUMN_CANDIDATES)

        for _, row in df.iterrows():
            raw_text = row.get(text_column, "")
            text = make_review_key(raw_text)
            if not text or len(text) < 10 or text in existing_keys:
                continue

            records.append(
                {
                    "source_file": file_path.name,
                    "title": normalize_spaces("" if name_column is None else str(row.get(name_column, ""))),
                    "data": normalize_spaces("" if date_column is None else str(row.get(date_column, ""))),
                    "teks_clean": text,
                }
            )

    if not records:
        return pd.DataFrame(columns=["source_file", "title", "data", "teks_clean"])

    candidates = pd.DataFrame(records)
    candidates = candidates.drop_duplicates(subset=["teks_clean"]).reset_index(drop=True)
    return candidates


def compute_negative_keyword_score(text: str) -> tuple[int, list[str], str]:
    total_score = 0
    hits = []
    for pattern, weight, tag in NEGATIVE_SIGNALS:
        found = re.search(pattern, text, flags=re.IGNORECASE)
        if found:
            total_score += weight
            hits.append(f"{tag}:{found.group(0)}")
    deduped = list(dict.fromkeys(hits))
    signal_level = "strong" if total_score >= 3 else "medium" if total_score >= 2 else "weak" if total_score >= 1 else "none"
    return total_score, deduped, signal_level


def select_negative_candidates(candidates: pd.DataFrame, model_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        empty = candidates.copy()
        return empty, empty

    model, vectorizer, metadata = load_artifacts(model_dir)
    preprocessor = IndonesianTextPreprocessor()
    predictions = predict_texts(model, vectorizer, preprocessor, candidates["teks_clean"].tolist())

    scored = candidates.copy()
    scored["prediksi_sentimen"] = predictions["prediksi_sentimen"]
    scored["ulasan_preprocessed"] = predictions["ulasan_preprocessed"]

    for column in predictions.columns:
        if column.startswith("prob_"):
            scored[column] = predictions[column]

    keyword_scores = scored["teks_clean"].map(compute_negative_keyword_score)
    scored["negative_keyword_score"] = keyword_scores.map(lambda item: item[0])
    scored["negative_keyword_hits"] = keyword_scores.map(lambda item: ", ".join(item[1]))
    scored["negative_signal_level"] = keyword_scores.map(lambda item: item[2])
    scored["selected_reason"] = ""

    model_negative = (scored["prediksi_sentimen"] == "negatif") & (scored["prob_negatif"] >= 0.40)
    keyword_negative = scored["negative_keyword_score"] >= 3
    mixed_negative = (scored["negative_keyword_score"] >= 2) & (scored["prob_negatif"] >= 0.15)
    selected_mask = model_negative | keyword_negative | mixed_negative

    scored.loc[model_negative, "selected_reason"] = "predicted_negative"
    scored.loc[keyword_negative & ~model_negative, "selected_reason"] = "keyword_negative"
    scored.loc[mixed_negative & ~(model_negative | keyword_negative), "selected_reason"] = "keyword_plus_probability"

    selected = scored[selected_mask].copy()
    selected["label"] = "negatif"

    rejected = scored[~selected_mask].copy()

    if metadata:
        scored["model_reference"] = metadata.get("best_model", "unknown")
        selected["model_reference"] = metadata.get("best_model", "unknown")
        rejected["model_reference"] = metadata.get("best_model", "unknown")

    return selected, rejected


def write_report(report_path: Path, selected: pd.DataFrame, rejected: pd.DataFrame) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        selected.to_excel(writer, index=False, sheet_name="selected_negative")
        rejected.to_excel(writer, index=False, sheet_name="rejected_candidates")


def append_to_dataset(data_path: Path, selected: pd.DataFrame) -> pd.DataFrame:
    existing = pd.read_excel(data_path)
    appended = selected[["title", "data", "teks_clean", "label"]].copy()
    updated = pd.concat([existing, appended], ignore_index=True)
    updated.to_excel(data_path, index=False)
    return updated


def main() -> None:
    args = parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {args.data_path.resolve()}")
    if not args.source_dir.exists():
        raise FileNotFoundError(f"Folder sumber tidak ditemukan: {args.source_dir.resolve()}")

    existing = pd.read_excel(args.data_path)
    existing_keys = set(existing["teks_clean"].fillna("").astype(str).map(make_review_key))

    candidates = extract_candidates(args.source_dir, args.pattern, existing_keys)
    selected, rejected = select_negative_candidates(candidates, args.model_dir)
    write_report(args.report_path, selected, rejected)

    print("=" * 70)
    print("AUGMENTASI DATASET")
    print("=" * 70)
    print(f"Dataset awal        : {len(existing)} baris")
    print(f"Kandidat baru       : {len(candidates)} baris")
    print(f"Terpilih negatif    : {len(selected)} baris")
    print(f"Report              : {args.report_path.resolve()}")
    if not selected.empty:
        preview = selected[
            [
                "source_file",
                "title",
                "teks_clean",
                "prediksi_sentimen",
                "prob_negatif",
                "negative_keyword_score",
                "selected_reason",
            ]
        ].head(10)
        print("-" * 70)
        print(preview.to_string(index=False))

    if args.dry_run:
        print("-" * 70)
        print("Dry run aktif. Dataset tidak diubah.")
        return

    updated = append_to_dataset(args.data_path, selected)
    print("-" * 70)
    print(f"Dataset akhir       : {len(updated)} baris")
    print(f"Simpan dataset      : {args.data_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()