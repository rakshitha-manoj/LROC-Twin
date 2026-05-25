"""
verify_dataset.py

Sanity checks on the processed JSONL splits.
Run after build_dataset.py to confirm the dataset is well-formed
before committing to a fine-tuning run.
"""

import json
import os
from pathlib import Path
from PIL import Image


PROCESSED_DIR = Path("data/processed")
SPLITS = ["lucid_train.jsonl", "lucid_val.jsonl", "lucid_test.jsonl"]


def verify_split(path: Path) -> dict:
    if not path.exists():
        print(f"[MISSING] {path}")
        return {}

    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))

    missing_images = 0
    broken_images = 0
    empty_responses = 0
    response_lengths = []

    for r in records:
        img_path = Path(r["image_path"])
        if not img_path.exists():
            missing_images += 1
        else:
            try:
                img = Image.open(img_path)
                img.verify()
            except Exception:
                broken_images += 1

        if not r.get("response", "").strip():
            empty_responses += 1

        response_lengths.append(len(r.get("response", "").split()))

    avg_len = sum(response_lengths) / len(response_lengths) if response_lengths else 0

    stats = {
        "total": len(records),
        "missing_images": missing_images,
        "broken_images": broken_images,
        "empty_responses": empty_responses,
        "avg_response_words": round(avg_len, 1),
    }

    status = "OK" if missing_images == 0 and broken_images == 0 and empty_responses == 0 else "ISSUES"
    print(f"[{status}] {path.name}: {stats}")
    return stats


def main():
    print("=== LROC-Twin: Dataset Verification ===\n")
    all_ok = True
    for split_name in SPLITS:
        stats = verify_split(PROCESSED_DIR / split_name)
        if stats.get("missing_images", 0) > 0 or stats.get("broken_images", 0) > 0:
            all_ok = False

    print()
    if all_ok:
        print("All splits verified. Ready for training.")
    else:
        print("Issues found. Fix before proceeding to Phase 2.")


if __name__ == "__main__":
    main()
