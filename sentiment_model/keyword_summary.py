from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

DEFAULT_STOPWORDS = {
    "dan",
    "di",
    "ke",
    "dari",
    "yang",
    "untuk",
    "pada",
    "dengan",
    "atau",
    "ini",
    "itu",
    "karena",
    "juga",
    "sangat",
    "ada",
    "tidak",
    "tempat",
    "wisata",
    "sana",
    "sini",
    "dalam",
    "sebagai",
    "sudah",
    "lagi",
    "saya",
    "kami",
    "kita",
}


def _tokenize(text: str, min_len: int = 3) -> list[str]:
    if not isinstance(text, str):
        return []
    return [tok for tok in text.split() if len(tok) >= min_len and tok not in DEFAULT_STOPWORDS]


def _top_keywords(tokens: list[str], top_n: int) -> list[dict[str, Any]]:
    counts = Counter(tokens)
    return [{"keyword": token, "count": count} for token, count in counts.most_common(top_n)]


def build_keyword_summary(
    reviews: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    top_n: int = 10,
) -> dict[str, Any]:
    overall_tokens: list[str] = []
    sentiment_tokens: dict[str, list[str]] = {"negative": [], "neutral": [], "positive": []}
    sentiment_counts: dict[str, int] = {"negative": 0, "neutral": 0, "positive": 0}

    destination_buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "review_count": 0,
            "sentiment_counts": {"negative": 0, "neutral": 0, "positive": 0},
            "tokens": [],
            "tokens_by_sentiment": {"negative": [], "neutral": [], "positive": []},
        }
    )

    for idx, row in enumerate(prediction_rows):
        review = reviews[idx] if idx < len(reviews) else {}
        destination_id = review.get("destination_id")
        if destination_id is None:
            destination_id = review.get("destination")
        destination_key = str(destination_id) if destination_id is not None else "unknown"

        sentiment = str(row.get("sentiment_label", row.get("label", ""))).lower()
        if sentiment not in sentiment_counts:
            sentiment = "neutral"

        processed_text = str(row.get("sentiment_processed_text", row.get("processed_text", "")) or "")
        tokens = _tokenize(processed_text)

        sentiment_counts[sentiment] += 1
        overall_tokens.extend(tokens)
        sentiment_tokens[sentiment].extend(tokens)

        bucket = destination_buckets[destination_key]
        bucket["review_count"] += 1
        bucket["sentiment_counts"][sentiment] += 1
        bucket["tokens"].extend(tokens)
        bucket["tokens_by_sentiment"][sentiment].extend(tokens)

    destinations: list[dict[str, Any]] = []
    for destination_id, bucket in destination_buckets.items():
        destinations.append(
            {
                "destination_id": destination_id,
                "review_count": bucket["review_count"],
                "sentiment_counts": bucket["sentiment_counts"],
                "top_keywords": _top_keywords(bucket["tokens"], top_n),
                "top_keywords_by_sentiment": {
                    "negative": _top_keywords(bucket["tokens_by_sentiment"]["negative"], top_n),
                    "neutral": _top_keywords(bucket["tokens_by_sentiment"]["neutral"], top_n),
                    "positive": _top_keywords(bucket["tokens_by_sentiment"]["positive"], top_n),
                },
            }
        )

    destinations.sort(key=lambda item: item["review_count"], reverse=True)

    return {
        "overall": {
            "review_count": len(prediction_rows),
            "sentiment_counts": sentiment_counts,
            "top_keywords": _top_keywords(overall_tokens, top_n),
            "top_keywords_by_sentiment": {
                "negative": _top_keywords(sentiment_tokens["negative"], top_n),
                "neutral": _top_keywords(sentiment_tokens["neutral"], top_n),
                "positive": _top_keywords(sentiment_tokens["positive"], top_n),
            },
        },
        "destinations": destinations,
    }
