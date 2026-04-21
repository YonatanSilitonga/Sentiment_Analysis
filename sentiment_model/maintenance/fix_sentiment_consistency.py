from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return [], "json-array"

    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("JSON root must be a list of objects.")
        return [item for item in parsed if isinstance(item, dict)], "json-array"

    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows, "ndjson"


def _dump_records(path: Path, rows: list[dict[str, Any]], fmt: str) -> None:
    if fmt == "ndjson":
        payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
        return

    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _fix_row(row: dict[str, Any], model_version: str | None) -> bool:
    label = row.get("sentiment_label")
    scores = row.get("sentiment_scores")
    if not isinstance(label, str) or not isinstance(scores, dict):
        return False
    if label not in scores:
        return False

    changed = False
    score_for_label = float(scores[label])
    current_conf = row.get("sentiment_confidence")
    if current_conf is None or abs(float(current_conf) - score_for_label) > 1e-9:
        row["sentiment_confidence"] = score_for_label
        changed = True

    row["sentiment_integrity"] = {
        "label_in_scores": True,
        "confidence_matches_label_score": True,
    }

    if model_version:
        if row.get("sentiment_model_version") != model_version:
            row["sentiment_model_version"] = model_version
            changed = True

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix sentiment confidence mismatches in JSON exports before re-importing to database."
    )
    parser.add_argument("--input", required=True, help="Input file path (.json array or .ndjson).")
    parser.add_argument("--output", help="Output file path. If omitted, use --inplace.")
    parser.add_argument("--inplace", action="store_true", help="Overwrite input file.")
    parser.add_argument("--model-version", default="", help="Optional model version to stamp into each record.")

    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not args.inplace and not args.output:
        raise ValueError("Provide --output or use --inplace.")

    rows, fmt = _load_records(input_path)

    checked = 0
    fixed = 0
    skipped = 0
    for row in rows:
        checked += 1
        if _fix_row(row, args.model_version.strip() or None):
            fixed += 1
        else:
            skipped += 1

    output_path = input_path if args.inplace else Path(args.output)
    _dump_records(output_path, rows, fmt)

    print("=== FIX SENTIMENT CONSISTENCY ===")
    print(f"Input       : {input_path}")
    print(f"Output      : {output_path}")
    print(f"Format      : {fmt}")
    print(f"Checked     : {checked}")
    print(f"Fixed       : {fixed}")
    print(f"Skipped     : {skipped}")


if __name__ == "__main__":
    main()
