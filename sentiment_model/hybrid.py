from __future__ import annotations

import re
from typing import Any

import pandas as pd

NEGATIVE_SIGNALS: list[tuple[str, int, str]] = [
    (r"\b(?:tidak|gak|ga|nggak|kurang)\s+bersih\b|\btidak\s+higienis\b", 3, "explicit_not_clean"),
    (r"\b(?:tidak|tak|gak|ga|nggak|kurang)(?:lah)?\s+(?:wangi|harum|segar)\b", 3, "negated_fragrance"),
    (r"\bkotor\b|\bjorok\b|\bjorok\s+parah\b|\bjijik\b|\bbau\b|\bpesing\b|\bapek\b|\banis\b|\bmengamis\b|\bmenyengat\b", 3, "kebersihan"),
    (r"\btoilet\s+(?:kotor|jorok|bau)\b|\bkamar\s+mandi\s+(?:kotor|bau)\b|\bwc\s+(?:kotor|jorok)\b|\bsampah\s+berserakan\b", 3, "dirty_toilet_facility"),
    (r"\bpungli\b|\bpungutan\s+liar\b|\bpungut(?:an)?\s+liar\b|\bcalo\b|\bdipalak\b|\bpemalak(?:an)?\b|\btipu\b|\bpenipuan\b", 3, "biaya_liar"),
    (r"\bparkir\s+liar\b|\bbiaya\s+liar\b|\btarif\s+liar\b|\bharga\s+liar\b", 3, "illegal_fee_practice"),
    (r"\bjutek\b|\bkasar\b|\btidak\s+ramah\b|\bgak\s+ramah\b|\btidak\s+profesional\b|\bgak\s+profesional\b|\blambat\b", 3, "pelayanan_buruk"),
    (r"\b(?:gak|ga|nggak|tidak|tak)\s+(?:rekomen|rekom|rekomendasi|recommended)\b", 3, "not_recommended"),
    (r"\bburuk\b|\bburuk\s+sekali\b|\bsangat\s+buruk\b|\bburuk\s+punya\b|\bburuk\s+berat\b", 3, "very_bad_service"),
    (r"\b(?:tidak|gak|ga|nggak)\s+ramah\b", 3, "not_friendly"),
    (r"\b(?:tidak|gak|ga|nggak)\s+profesional\b", 3, "not_professional"),
    (r"\b(?:tidak|gak|ga|nggak)\s+sopan\b", 3, "not_polite"),
    (r"\brusak\b|\bberlubang\b|\bsempit\b|\bmacet\b|\bnyasar\b", 2, "akses_sulit"),
    (r"\bakses\s+sulit\b|\bsusah\s+akses\b|\bsulit\s+dijangkau\b|\bjalan\s+rusak\b|\bjalan\s+jelek\b", 2, "hard_access"),
    (r"\bmedan\s+berbahaya\b|\bmedan\s+sulit\b|\bjalan\s+curam\b|\btanjakan\s+curam\b", 2, "dangerous_terrain"),
    (r"\bnggak\s+ada\b|\btidak\s+ada\b|\bnggak\s+layak\b|\btidak\s+layak\b|\bterbengkalai\b|\bbocor\b", 2, "fasilitas_rusak"),
    (r"\bbahaya\b|\bnggak\s+aman\b|\btidak\s+aman\b|\bpenipuan\b|\bditipu\b|\bcopet\b", 2, "keamanan_terganggu"),
    (r"\b(?:tidak|gak|ga|nggak)\s+aman\b|\bgak amanah\b|\btidak amanah\b", 3, "unsafe_or_untrustworthy"),
    (r"\b(?:tidak|gak|ga|nggak|tak)\s+boleh\b", 3, "not_allowed"),
    (r"\b(?:dipulang(?:kan)?|disuruh pulang|ditolak)\b", 3, "turned_away"),
    (r"\btidak memuaskan\b|\bkurang memuaskan\b|\bkecewa\b|\bkecewa\s+berat\b|\bsangat\s+kecewa\b|\bmengecewakan\b|\bmengecewakan\s+sekali\b", 3, "unsatisfactory"),
    (r"\b(?:gak|ga|nggak|tidak)\s+akan\s+kesini\s+lagi\b", 3, "wont_return"),
    (r"\bmaling\b|\bcuri\b", 3, "theft_risk"),
    (r"\bbohong\b|\bdibohongi\b", 3, "deception"),
    (r"\bantre(?:an)?\s+panjang\b|\bantri\s+panjang\b", 1, "long_queue"),
    (r"\bparah\b", 2, "bad_experience"),
    (r"\bsombong\b|\bsewot\b", 2, "bad_attitude"),
    (r"\bmahal\b|\bgak masuk akal\b|\btidak masuk akal\b|\boverprice\b", 2, "overpriced"),
    (r"\bgembok\b|\bdigembok\b", 2, "locked_facility"),
    (r"\bberisik\b|\bribut\b", 1, "noisy_environment"),
]

POSITIVE_SIGNALS: list[tuple[str, int, str]] = [
    (r"\bbersih\b", 2, "cleanliness_positive"),
    (r"\bwangi\b|\bharum\b|\bsegar\b", 2, "fragrance_positive"),
    (r"\bnyaman\b|\baman\b|\btenang\b", 2, "comfort_safety"),
    (r"\bbagus\b|\bindah\b|\brapi\b|\bcantik\b", 2, "quality_visual"),
    (r"\bramah\b|\bsopan\b|\bhelpful\b|\bcepat\b", 1, "service_positive"),
    (r"\bpuas\b|\bmemuaskan\b|\brekomendasi\b|\brekomen\b", 2, "satisfaction"),
]

CONTRAST_SIGNALS: list[tuple[str, int, str]] = [
    (
        r"\btapi\b|\btetapi\b|\bnamun\b|\bakan\s+tetapi\b|"
        r"\bwalaupun\b|\bwalau\b|\bmeskipun\b|\bmeski\b|\bbiarpun\b|\bkendati(?:pun)?\b|"
        r"\bhanya\s+saja\b|\bcuma(?:n)?\b|"
        r"\bpadahal\b|\bsedangkan\b|\bsebaliknya\b|\bdi\s+sisi\s+lain\b|\bsementara\s+itu\b",
        1,
        "contrast_connector",
    ),
    (r"\blama\b|\bpanjang\b|\bjauh\b|\bcapek\b|\blelah\b", 1, "mild_travel_discomfort"),
]


def default_hybrid_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "negative_label": "negatif",
        "high_confidence_threshold": 0.55,
        "mid_confidence_threshold": 0.30,
        "mid_keyword_score_threshold": 2,
        "strong_signal_score_threshold": 3,
        "strong_signal_min_prob": 0.15,
        "positive_override_enabled": False,
        "positive_label": "positif",
        "positive_keyword_score_threshold": 3,
        "positive_mid_confidence_threshold": 0.30,
        "positive_block_negative_prob": 0.65,
        "neutral_override_enabled": False,
        "neutral_label": "netral",
        "neutral_min_positive_score": 2,
        "neutral_max_negative_score": 1,
        "neutral_max_negative_prob": 0.35,
    }


def compute_negative_keyword_score(text: str) -> tuple[int, list[str], str]:
    total_score = 0
    hits: list[str] = []
    for pattern, weight, tag in NEGATIVE_SIGNALS:
        found = re.search(pattern, text, flags=re.IGNORECASE)
        if found:
            total_score += weight
            hits.append(f"{tag}:{found.group(0)}")

    deduped_hits = list(dict.fromkeys(hits))
    signal_level = "strong" if total_score >= 3 else "medium" if total_score >= 2 else "weak" if total_score >= 1 else "none"
    return total_score, deduped_hits, signal_level


def compute_positive_keyword_score(text: str) -> tuple[int, list[str], str]:
    total_score = 0
    hits: list[str] = []
    for pattern, weight, tag in POSITIVE_SIGNALS:
        found = re.search(pattern, text, flags=re.IGNORECASE)
        if found:
            total_score += weight
            hits.append(f"{tag}:{found.group(0)}")

    deduped_hits = list(dict.fromkeys(hits))
    signal_level = "strong" if total_score >= 3 else "medium" if total_score >= 2 else "weak" if total_score >= 1 else "none"
    return total_score, deduped_hits, signal_level


def compute_contrast_keyword_score(text: str) -> tuple[int, list[str], str]:
    total_score = 0
    hits: list[str] = []
    for pattern, weight, tag in CONTRAST_SIGNALS:
        found = re.search(pattern, text, flags=re.IGNORECASE)
        if found:
            total_score += weight
            hits.append(f"{tag}:{found.group(0)}")

    deduped_hits = list(dict.fromkeys(hits))
    signal_level = "strong" if total_score >= 2 else "weak" if total_score >= 1 else "none"
    return total_score, deduped_hits, signal_level


def apply_hybrid_rules(
    result_df: pd.DataFrame,
    original_texts: list[str],
    hybrid_config: dict[str, Any] | None,
) -> pd.DataFrame:
    if not hybrid_config or not hybrid_config.get("enabled", False):
        return result_df

    negative_label = str(hybrid_config.get("negative_label", "negatif"))
    prob_column = f"prob_{negative_label}"

    if prob_column not in result_df.columns:
        return result_df

    high_threshold = float(hybrid_config.get("high_confidence_threshold", 0.55))
    mid_threshold = float(hybrid_config.get("mid_confidence_threshold", 0.30))
    mid_score_threshold = int(hybrid_config.get("mid_keyword_score_threshold", 2))
    strong_score_threshold = int(hybrid_config.get("strong_signal_score_threshold", 3))
    strong_min_prob = float(hybrid_config.get("strong_signal_min_prob", 0.15))

    positive_override_enabled = bool(hybrid_config.get("positive_override_enabled", False))
    positive_label = str(hybrid_config.get("positive_label", "positif"))
    positive_prob_column = f"prob_{positive_label}"
    positive_score_threshold = int(hybrid_config.get("positive_keyword_score_threshold", 3))
    positive_mid_prob_threshold = float(hybrid_config.get("positive_mid_confidence_threshold", 0.30))
    positive_block_negative_prob = float(hybrid_config.get("positive_block_negative_prob", 0.65))

    neutral_override_enabled = bool(hybrid_config.get("neutral_override_enabled", False))
    neutral_label = str(hybrid_config.get("neutral_label", "netral"))
    neutral_min_positive_score = int(hybrid_config.get("neutral_min_positive_score", 2))
    neutral_max_negative_score = int(hybrid_config.get("neutral_max_negative_score", 1))
    neutral_max_negative_prob = float(hybrid_config.get("neutral_max_negative_prob", 0.35))

    output_df = result_df.copy()
    output_df["prediksi_sentimen_base"] = output_df["prediksi_sentimen"]
    output_df["negative_keyword_score"] = 0
    output_df["negative_keyword_hits"] = ""
    output_df["negative_signal_level"] = "none"
    output_df["positive_keyword_score"] = 0
    output_df["positive_keyword_hits"] = ""
    output_df["positive_signal_level"] = "none"
    output_df["contrast_keyword_score"] = 0
    output_df["contrast_keyword_hits"] = ""
    output_df["contrast_signal_level"] = "none"
    output_df["hybrid_reason"] = "model_only"

    for idx, raw_text in enumerate(original_texts):
        score_neg, hits_neg, level_neg = compute_negative_keyword_score(str(raw_text))
        score_pos, hits_pos, level_pos = compute_positive_keyword_score(str(raw_text))
        score_contrast, hits_contrast, level_contrast = compute_contrast_keyword_score(str(raw_text))

        prob_neg = float(output_df.at[idx, prob_column])
        prob_pos = float(output_df.at[idx, positive_prob_column]) if positive_prob_column in output_df.columns else 0.0

        final_label = str(output_df.at[idx, "prediksi_sentimen_base"])
        reason = "model_only"

        if prob_neg >= high_threshold:
            final_label = negative_label
            reason = "high_confidence_negative"
        elif prob_neg >= mid_threshold and score_neg >= mid_score_threshold:
            final_label = negative_label
            reason = "mid_confidence_plus_keywords"
        elif score_neg >= strong_score_threshold and prob_neg >= strong_min_prob:
            final_label = negative_label
            reason = "strong_negative_keywords"

        if (
            positive_override_enabled
            and score_neg == 0
            and score_pos >= positive_score_threshold
            and prob_pos >= positive_mid_prob_threshold
            and prob_neg < positive_block_negative_prob
            and prob_pos >= prob_neg
        ):
            final_label = positive_label
            reason = "positive_keywords_override"

        if (
            neutral_override_enabled
            and score_contrast >= 2
            and score_neg <= neutral_max_negative_score
            and prob_neg <= neutral_max_negative_prob
            and (
                score_pos >= neutral_min_positive_score
                or final_label == positive_label
            )
        ):
            final_label = neutral_label
            reason = "contrast_to_neutral_override"

        output_df.at[idx, "negative_keyword_score"] = score_neg
        output_df.at[idx, "negative_keyword_hits"] = ", ".join(hits_neg)
        output_df.at[idx, "negative_signal_level"] = level_neg
        output_df.at[idx, "positive_keyword_score"] = score_pos
        output_df.at[idx, "positive_keyword_hits"] = ", ".join(hits_pos)
        output_df.at[idx, "positive_signal_level"] = level_pos
        output_df.at[idx, "contrast_keyword_score"] = score_contrast
        output_df.at[idx, "contrast_keyword_hits"] = ", ".join(hits_contrast)
        output_df.at[idx, "contrast_signal_level"] = level_contrast
        output_df.at[idx, "prediksi_sentimen"] = final_label
        output_df.at[idx, "hybrid_reason"] = reason

    return output_df
