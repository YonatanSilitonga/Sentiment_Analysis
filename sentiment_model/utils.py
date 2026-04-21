from __future__ import annotations

import re

import pandas as pd

TEXT_COLUMN_CANDIDATES = [
    "ulasan",
    "review",
    "text",
    "teks_clean",
    "teks_asli",
    "komentar",
    "data5",
    "data2",
]


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered_map = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in lowered_map:
            return lowered_map[candidate]
    return None


def score_text_column(series: pd.Series) -> float:
    texts = series.dropna().astype(str).map(normalize_spaces)
    texts = texts[texts != ""]
    if texts.empty:
        return float("-inf")

    mean_len = texts.str.len().mean()
    mean_words = texts.str.split().map(len).mean()
    alpha_ratio = texts.map(
        lambda value: sum(ch.isalpha() for ch in value) / max(len(value), 1)
    ).mean()
    url_ratio = texts.str.contains(
        r"https?://|www\\.|googleusercontent", case=False, regex=True
    ).mean()
    ui_ratio = texts.str.contains(r"Bagikan|Suka|foto|ulasan", case=False, regex=True).mean()

    return (mean_len * 1.0) + (mean_words * 4.0) + (alpha_ratio * 40.0) - (url_ratio * 100.0) - (ui_ratio * 25.0)


def choose_text_column(
    df: pd.DataFrame,
    explicit_column: str | None = None,
    candidates: list[str] | None = None,
    metadata_columns: set[str] | None = None,
    use_scoring_fallback: bool = False,
) -> str:
    if explicit_column:
        if explicit_column not in df.columns:
            raise ValueError(f"Kolom `{explicit_column}` tidak ditemukan di input file.")
        return explicit_column

    if candidates is None:
        candidates = TEXT_COLUMN_CANDIDATES

    explicit = find_column(df, candidates)
    if explicit is not None:
        return explicit

    if use_scoring_fallback:
        excluded = metadata_columns or set()
        text_like_columns: list[tuple[str, float]] = []

        for column in df.columns:
            normalized_name = str(column).strip().lower()
            if normalized_name in excluded:
                continue

            series = df[column]
            if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
                continue

            text_like_columns.append((column, score_text_column(series)))

        if not text_like_columns:
            raise ValueError(f"Tidak menemukan kolom teks pada file dengan kolom: {list(df.columns)}")

        text_like_columns.sort(key=lambda item: item[1], reverse=True)
        return text_like_columns[0][0]

    object_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if object_columns:
        return object_columns[0]

    raise ValueError("Tidak ada kolom teks yang bisa dipakai untuk prediksi batch.")