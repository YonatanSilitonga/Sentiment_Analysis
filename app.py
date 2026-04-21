from __future__ import annotations

import os
from functools import wraps
from typing import Any, Callable

from flask import Flask, jsonify, request
from flask_cors import CORS

from sentiment_model.keyword_summary import build_keyword_summary
from sentiment_model.sentiment_model import SentimentAnalysisService


def _json_error(message: str, status_code: int = 400, error_type: str = "bad_request", details: Any | None = None):
    payload = {
        "success": False,
        "error": {
            "type": error_type,
            "message": message,
        },
    }
    if details is not None:
        payload["error"]["details"] = details
    return jsonify(payload), status_code


def _handle_exceptions(fn: Callable):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as exc:
            return _json_error(str(exc), 400, "validation_error")
        except FileNotFoundError as exc:
            return _json_error(str(exc), 500, "artifact_not_found")
        except Exception as exc:  # pragma: no cover - defensive guard for runtime API
            return _json_error(f"Internal server error: {exc}", 500, "internal_error")

    return wrapper


def create_app() -> Flask:
    app = Flask(__name__)
    if os.getenv("PY_SENTIMENT_ENABLE_CORS", "true").lower() in {"1", "true", "yes", "on"}:
        CORS(app)

    service = SentimentAnalysisService(
        model_dir=os.getenv("PY_SENTIMENT_MODEL_DIR", "sentiment_model/model_artifacts_v9"),
        long_text_mode=os.getenv("PY_SENTIMENT_LONG_TEXT_MODE", "auto"),
        long_text_min_words=int(os.getenv("PY_SENTIMENT_LONG_TEXT_MIN_WORDS", "14")),
        long_text_min_segments=int(os.getenv("PY_SENTIMENT_LONG_TEXT_MIN_SEGMENTS", "2")),
        close_margin_neutral=float(os.getenv("PY_SENTIMENT_CLOSE_MARGIN_NEUTRAL", "0.06")),
    )
    app.config["sentiment_service"] = service

    @app.get("/health")
    @_handle_exceptions
    def health():
        return jsonify(service.health())

    @app.post("/api/v1/predict")
    @_handle_exceptions
    def predict():
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return _json_error("Request body must be a JSON object.", 400, "validation_error")
        text = payload.get("text")
        review_id = payload.get("review_id")
        return jsonify(service.predict_single(text=text, review_id=review_id))

    @app.post("/api/v1/predict-batch")
    @_handle_exceptions
    def predict_batch():
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return _json_error("Request body must be a JSON object.", 400, "validation_error")
        reviews = payload.get("reviews")
        if reviews is None:
            reviews = payload.get("items")
        if reviews is None:
            reviews = payload.get("data")
        include_sentence_predictions = bool(payload.get("include_sentence_predictions", False))
        return jsonify(service.predict_batch(reviews=reviews, include_sentence_predictions=include_sentence_predictions))

    @app.post("/api/v1/summary-keywords")
    @_handle_exceptions
    def summary_keywords():
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return _json_error("Request body must be a JSON object.", 400, "validation_error")

        reviews = payload.get("reviews")
        if reviews is None:
            reviews = payload.get("items")
        if reviews is None:
            reviews = payload.get("data")

        if not isinstance(reviews, list) or not reviews:
            return _json_error("Field 'reviews' wajib berupa array yang tidak kosong.", 400, "validation_error")

        top_n = int(payload.get("top_n", 10))
        if top_n < 1:
            top_n = 1
        if top_n > 50:
            top_n = 50

        include_sentence_predictions = bool(payload.get("include_sentence_predictions", False))
        prediction_result = service.predict_batch(reviews=reviews, include_sentence_predictions=include_sentence_predictions)
        prediction_rows = prediction_result.get("data", {}).get("results", [])
        keyword_summary = build_keyword_summary(reviews=reviews, prediction_rows=prediction_rows, top_n=top_n)

        return jsonify(
            {
                "success": True,
                "data": {
                    "model_version": service.model_version,
                    "top_n": top_n,
                    "prediction_summary": prediction_result.get("data", {}).get("summary", {}),
                    "keyword_summary": keyword_summary,
                },
            }
        )

    @app.get("/api/v1/stats")
    @_handle_exceptions
    def stats():
        return jsonify(service.stats())

    @app.get("/")
    def index():
        return jsonify(
            {
                "success": True,
                "data": {
                    "service": "sentiment-analysis-api",
                    "model_version": service.model_version,
                    "endpoints": [
                        "/health",
                        "/api/v1/predict",
                        "/api/v1/predict-batch",
                        "/api/v1/summary-keywords",
                        "/api/v1/stats",
                    ],
                },
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("PY_SENTIMENT_HOST", "127.0.0.1"),
        port=int(os.getenv("PY_SENTIMENT_PORT", "5000")),
        debug=os.getenv("PY_SENTIMENT_DEBUG", "false").lower() in {"1", "true", "yes", "on"},
    )
