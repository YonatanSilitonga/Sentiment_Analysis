# Integration Guide - Python Sentiment Service

## Overview

This project exposes a Flask-based sentiment API that is consumed by a Laravel admin panel. The Python service loads the trained joblib artifact once at startup and returns normalized JSON responses for single and batch review analysis.

## Project Structure

- `app.py` - Flask API entrypoint.
- `sentiment_model/sentiment_model.py` - reusable inference service wrapper.
- `sentiment_model/preprocessing.py` - text cleaning and normalization.
- `sentiment_model/hybrid.py` - hybrid rule engine.
- `sentiment_model/long_text.py` - long-text segmentation and aggregation.
- `sentiment_model/model_artifacts_v8/` - trained model, vectorizer, metadata, and hybrid rules.
- `test_setup.py` - environment and smoke-test script.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the API

```bash
set PY_SENTIMENT_PORT=5000
python app.py
```

Optional environment variables:

- `PY_SENTIMENT_MODEL_DIR` - default `sentiment_model/model_artifacts_v8`
- `PY_SENTIMENT_HOST` - default `127.0.0.1`
- `PY_SENTIMENT_PORT` - default `5000`
- `PY_SENTIMENT_DEBUG` - default `false`
- `PY_SENTIMENT_ENABLE_CORS` - default `true`
- `PY_SENTIMENT_LONG_TEXT_MODE` - default `auto`
- `PY_SENTIMENT_LONG_TEXT_MIN_WORDS` - default `14`
- `PY_SENTIMENT_LONG_TEXT_MIN_SEGMENTS` - default `2`
- `PY_SENTIMENT_CLOSE_MARGIN_NEUTRAL` - default `0.06`

## Endpoints

### `GET /health`

Returns service readiness and model version.

### `POST /api/v1/predict`

Request body:

```json
{
  "review_id": 123,
  "text": "Tempatnya bagus tapi toiletnya kurang bersih."
}
```

Response:

```json
{
  "success": true,
  "data": {
    "label": "neutral",
    "confidence": 0.66,
    "scores": {
      "negative": 0.20,
      "neutral": 0.66,
      "positive": 0.14
    },
    "reason": "mixed_polarity_conflict_to_neutral",
    "processed_text": "tempat bagus toilet kurang bersih",
    "review_id": 123,
    "long_text_used": false,
    "long_text_reason": "short_text_default",
    "uncertainty_margin": 0.12
  }
}
```

### `POST /api/v1/predict-batch`

Request body:

```json
{
  "reviews": [
    { "id": 1, "text": "Bagus sekali." },
    { "id": 2, "text": "Toiletnya kotor dan bau." }
  ]
}
```

### `GET /api/v1/stats`

Returns model version, dataset source, and training metrics.

## Laravel Request Example

```php
Http::timeout(5)->post(env('PY_SENTIMENT_URL') . '/api/v1/predict', [
    'review_id' => $review->id,
    'text' => $review->comment,
]);
```

## Troubleshooting

1. `ModuleNotFoundError: Sastrawi` - run `pip install -r requirements.txt` in the active virtualenv.
2. `FileNotFoundError` for model artifacts - verify `sentiment_model/model_artifacts_v8/` exists.
3. API returns 500 on startup - run `python test_setup.py` to isolate dependency or artifact issues.
4. Batch request fails - ensure payload uses `reviews` as an array of objects with `id` and `text`.
