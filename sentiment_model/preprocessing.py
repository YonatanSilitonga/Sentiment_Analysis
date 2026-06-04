from __future__ import annotations

import re
from dataclasses import dataclass

import json
from pathlib import Path

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

LABEL_ORDER = ["negatif", "netral", "positif"]

COLUMN_ALIASES = {
    "title": "nama",
    "nama": "nama",
    "name": "nama",
    "data": "tanggal",
    "tanggal": "tanggal",
    "date": "tanggal",
    "teks_clean": "ulasan",
    "teks_asli": "ulasan",
    "ulasan": "ulasan",
    "review": "ulasan",
    "text": "ulasan",
    "komentar": "ulasan",
    "label": "label",
    "sentimen": "label",
    "sentiment": "label",
}

REQUIRED_COLUMNS = ["nama", "tanggal", "ulasan", "label"]

NORMALIZATION_PATTERNS = [
    (r"\b(?:gk|ga|gak|nggak)\b", " tidak "),
    (r"\bblm\b", " belum "),
    (r"\bbgt\b|\bbanget\b|\bbangat\b", " sangat "),
    (r"\b(?:rekomen|rekomended|recomended|recommended|rekomended|recommend)\b", " rekomendasi "),
    (r"\bkrn\b", " karena "),
    (r"\btp\b", " tapi "),
    (r"\bjg\b", " juga "),
]

PHRASE_NORMALIZATION_PATTERNS = [
    (r"\btidak\s+ada\s+kendala\b", " tidak_ada_kendala "),
    (r"\btidak\s+ada\s+masalah\b", " tidak_ada_masalah "),
    (r"\btidak\s+memuaskan\b", " tidak_memuaskan "),
    (r"\bkurang\s+memuaskan\b", " kurang_memuaskan "),
    (r"\bsangat\s+mengecewakan\b", " sangat_mengecewakan "),
    (r"\bmengecewakan\s+sekali\b", " mengecewakan_sekali "),
    (r"\bburuk\s+sekali\b", " buruk_sekali "),
    (r"\bsangat\s+buruk\b", " sangat_buruk "),
    (r"\bsangat\s+bagus\b", " sangat_bagus "),
    (r"\bsangat\s+memuaskan\b", " sangat_memuaskan "),
    (r"\bsangat\s+nyaman\b", " sangat_nyaman "),
    (r"\bsangat\s+ramah\b", " sangat_ramah "),
    (r"\btidak\s+tembus\s+suara\b", " tidak_tembus_suara "),
    (r"\bkedap\s+suara\b", " kedap_suara "),
    (r"\bsesuai\s+harga\b|\bsesuai\s+tarif\b|\bsesuai\s+budget\b", " sesuai_harga "),
    (r"\bbiasa\s+aja\b", " biasa_saja "),
]

PROTECTED_TOKENS = {
    "pengalaman",
    "pelayanan",
    "buruk",
    "kecewa",
    "mengecewakan",
    "memuaskan",
    "ramah",
    "bersih",
    "kotor",
    "nyaman",
    "aman",
    "bagus",
    "indah",
    "jelek",
    "lambat",
    "cepat",
    "mahal",
    "murah",
    "tidak_ada_kendala",
    "tidak_ada_masalah",
    "tidak_memuaskan",
    "kurang_memuaskan",
    "sangat_mengecewakan",
    "mengecewakan_sekali",
    "buruk_sekali",
    "sangat_buruk",
    "sangat_bagus",
    "sangat_memuaskan",
    "sangat_nyaman",
    "sangat_ramah",
    "tidak_tembus_suara",
    "kedap_suara",
    "sesuai_harga",
    "biasa_saja",
}


def strip_edge_ellipsis(text: str) -> str:
    text = re.sub(r"^\s*(?:\.{2,}|…)+\s*", "", text)
    text = re.sub(r"\s*(?:\.{2,}|…)+\s*$", "", text)
    return text.strip()


def prepare_labeled_dataframe(raw_df: pd.DataFrame, label_order: list[str] | None = None) -> pd.DataFrame:
    if label_order is None:
        label_order = LABEL_ORDER

    df = raw_df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    df = df.rename(columns={col: COLUMN_ALIASES.get(col, col) for col in df.columns})

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {missing_columns}. "
            f"Kolom yang tersedia: {list(raw_df.columns)}"
        )

    df = df[REQUIRED_COLUMNS].copy()
    df["nama"] = df["nama"].fillna("").astype(str).str.strip()
    df["tanggal"] = df["tanggal"].fillna("").astype(str).str.strip()
    df["ulasan"] = df["ulasan"].fillna("").astype(str).str.strip().apply(strip_edge_ellipsis)
    df["label"] = df["label"].fillna("").astype(str).str.strip().str.lower()

    valid_labels = set(label_order)
    df = df[(df["ulasan"] != "") & (df["label"].isin(valid_labels))].reset_index(drop=True)
    return df


@dataclass
class IndonesianTextPreprocessor:
    """Pipeline preprocessing teks Bahasa Indonesia untuk sentimen."""

    keep_negation: bool = True

    def __post_init__(self) -> None:
        self.stemmer = StemmerFactory().create_stemmer()
        stop_factory = StopWordRemoverFactory()
        self.stopwords_id = set(stop_factory.get_stop_words())

        if self.keep_negation:
            negation_words = {
                "tidak", "bukan", "jangan", "belum", "kurang", "ga", "gak", "nggak", "tak"
            }
            self.stopwords_id -= negation_words

        custom_stopwords = {"yg", "aja", "nya", "nih", "sih", "dong", "deh"}
        self.stopwords_id |= custom_stopwords

        # Load slang dictionary
        self.slang_dict: dict[str, str] = {}
        try:
            project_root = Path(__file__).resolve().parent.parent
            slang_path = project_root / "combined_slang_words.txt"
            if slang_path.exists():
                with open(slang_path, "r", encoding="utf-8") as f:
                    self.slang_dict = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load slang dictionary: {e}")

    def normalize_text(self, text: str) -> str:
        normalized = "" if pd.isna(text) else str(text)
        normalized = strip_edge_ellipsis(normalized)
        # Fix missing space after punctuation (e.g. "disini.dan" -> "disini. dan")
        normalized = re.sub(r"(?<=[.!?,;])(?=[^\s\d])", r" ", normalized)
        normalized = re.sub(r"https?://\S+|www\.\S+", " ", normalized)
        for pattern, replacement in NORMALIZATION_PATTERNS:
            normalized = re.sub(pattern, replacement, normalized)
        # Fix missing spaces in camelCase-like strings (e.g., hotelPelayanan -> hotel Pelayanan)
        normalized = re.sub(r"([a-z])([A-Z])", r"\1 \2", normalized)
        # Fix reduplication with '2' (e.g., marah2 -> marah marah)
        normalized = re.sub(r"\b(\w+)2\b", r"\1 \1", normalized)
        normalized = normalized.lower()
        for pattern, replacement in PHRASE_NORMALIZATION_PATTERNS:
            normalized = re.sub(pattern, replacement, normalized)
        normalized = re.sub(r"[^a-zA-Z_\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def tokenize_text(text: str) -> list[str]:
        # Normalisasi pengulangan huruf berlebih (misal: baguuuusss -> bagus)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # Tokenisasi dasar
        return [token for token in text.split() if token]

    def normalize_slang(self, tokens: list[str]) -> list[str]:
        if not self.slang_dict:
            return tokens
        # Split mapped words because some slang maps to multiple words (e.g. "gaada" -> "tidak ada uang")
        normalized_tokens = []
        for token in tokens:
            mapped = self.slang_dict.get(token, token)
            normalized_tokens.extend(mapped.split())
        return normalized_tokens

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        return [token for token in tokens if token not in self.stopwords_id and len(token) > 1]

    def stem_text(self, tokens: list[str]) -> str:
        processed_tokens = []
        placeholders = {}
        for idx, token in enumerate(tokens):
            if token in PROTECTED_TOKENS:
                placeholder = f"protectedtoken{idx}"
                placeholders[placeholder] = token
                processed_tokens.append(placeholder)
            else:
                processed_tokens.append(token)

        stemmed_text = self.stemmer.stem(" ".join(processed_tokens)) if processed_tokens else ""
        
        final_tokens = []
        for token in stemmed_text.split():
            orig_token = placeholders.get(token, token)
            final_tokens.append(orig_token)
            
        return " ".join(final_tokens)

    def preprocess_text(self, text: str) -> str:
        normalized = self.normalize_text(text)
        tokens = self.tokenize_text(normalized)
        tokens = self.normalize_slang(tokens)
        filtered_tokens = self.remove_stopwords(tokens)
        return self.stem_text(filtered_tokens)
