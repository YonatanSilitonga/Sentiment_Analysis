from __future__ import annotations

import importlib
from pathlib import Path

from app import create_app
from sentiment_model.sentiment_model import SentimentAnalysisService


REQUIRED_MODULES = [
    "flask",
    "flask_cors",
    "joblib",
    "numpy",
    "pandas",
    "sklearn",
    "Sastrawi",
]


def check_imports() -> None:
    print("== Import check ==")
    for module_name in REQUIRED_MODULES:
        module = importlib.import_module(module_name)
        print(f"OK  {module_name:<12} -> {getattr(module, '__file__', 'builtin')}")


def check_service() -> SentimentAnalysisService:
    print("\n== Service init ==")
    service = SentimentAnalysisService(model_dir=Path("sentiment_model/model_artifacts_v8"))
    print(f"OK  model_version  -> {service.model_version}")
    print(f"OK  model_dir      -> {service.model_dir.resolve()}")
    return service


def check_prediction(service: SentimentAnalysisService) -> None:
    print("\n== Sample prediction ==")
    sample = "Pemandangannya indah tetapi toiletnya kurang bersih dan baunya menyengat."
    result = service.predict_single(sample, review_id=1)
    data = result["data"]
    print(f"OK  label          -> {data['label']}")
    print(f"OK  confidence     -> {data['confidence']}")
    print(f"OK  reason         -> {data['reason']}")
    print(f"OK  processed_text -> {data['processed_text']}")


def check_flask_app() -> None:
    print("\n== Flask app smoke test ==")
    app = create_app()
    with app.test_client() as client:
        health = client.get("/health")
        print(f"OK  /health status -> {health.status_code}")
        print(f"OK  /health body   -> {health.get_json()}")


def main() -> None:
    check_imports()
    service = check_service()
    check_prediction(service)
    check_flask_app()
    print("\nAll setup checks passed.")


if __name__ == "__main__":
    main()
