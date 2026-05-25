"""
build_dataset.py

Downloads pcvlab/lucid (stage2_qa split) from HuggingFace,
converts multi-turn LLaVA conversations into instruction-tuning
triples, saves images locally, and writes JSONL splits.

Usage:
    python training/scripts/build_dataset.py [--max-samples N]

Arguments:
    --max-samples   Cap on total samples to process (default: 2000).
                    Set to 0 for the full 20K dataset.

Output:
    data/lucid/images/          PNG files saved locally
    data/processed/lucid_train.jsonl
    data/processed/lucid_val.jsonl
    data/processed/lucid_test.jsonl
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

SEED = 42
TRAIN_RATIO = 0.75
VAL_RATIO = 0.10
TEST_RATIO = 0.15

HF_DATASET_ID = "pcvlab/lucid"
HF_CONFIG = "stage2_qa"

IMAGE_DIR = Path("data/lucid/images")
OUTPUT_DIR = Path("data/processed")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a lunar surface analyst assisting an autonomous rover navigation system. "
    "Analyze the provided lunar terrain image carefully. "
    "Describe surface features, identify any hazards or obstacles relevant to rover traversal, "
    "and provide a navigation recommendation where appropriate."
)

random.seed(SEED)


def sanitize_filename(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace(".", "_")


def extract_conversation(conversations: list[dict]) -> tuple[str, str]:
    """
    Converts a LLaVA-format multi-turn conversation into a single
    prompt and response string.

    Human turns are concatenated (with <image> token stripped) to form
    the prompt. GPT turns are concatenated to form the response.
    The system prompt is prepended to the human side.
    """
    human_turns = []
    gpt_turns = []

    for turn in conversations:
        role = turn.get("from", "")
        value = turn.get("value", "").replace("<image>", "").strip()
        if not value:
            continue
        if role == "human":
            human_turns.append(value)
        elif role == "gpt":
            gpt_turns.append(value)

    prompt = SYSTEM_PROMPT + "\n\n" + "\n".join(human_turns)
    response = "\n".join(gpt_turns)

    return prompt.strip(), response.strip()


def save_image(pil_image: Image.Image, sample_id: str) -> str:
    """Saves a PIL image to disk and returns the local path."""
    filename = sanitize_filename(sample_id) + ".png"
    path = IMAGE_DIR / filename
    if not path.exists():
        pil_image.convert("RGB").save(path, format="PNG")
    return str(path)


def process_dataset(max_samples: int) -> list[dict]:
    print(f"Loading {HF_DATASET_ID} / {HF_CONFIG} from HuggingFace...")
    print("Using streaming to avoid full download upfront.\n")

    dataset = load_dataset(
        HF_DATASET_ID,
        HF_CONFIG,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    samples = []
    cap = max_samples if max_samples > 0 else float("inf")

    for record in tqdm(dataset, desc="Processing LUCID samples", total=max_samples or None):
        if len(samples) >= cap:
            break

        sample_id = record.get("id", f"sample_{len(samples):05d}")
        conversations = record.get("conversations", [])
        pil_image = record.get("image")

        if not conversations or pil_image is None:
            continue

        prompt, response = extract_conversation(conversations)

        if not prompt or not response:
            continue

        image_path = save_image(pil_image, sample_id)

        samples.append({
            "id": sample_id,
            "image_path": image_path,
            "prompt": prompt,
            "response": response,
            "source": "lucid",
        })

    print(f"\nProcessed {len(samples)} valid samples.")
    return samples


def split_and_write(samples: list[dict]) -> None:
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }

    for split_name, split_samples in splits.items():
        for s in split_samples:
            s["split"] = split_name

        out_path = OUTPUT_DIR / f"lucid_{split_name}.jsonl"
        with open(out_path, "w") as f:
            for s in split_samples:
                f.write(json.dumps(s) + "\n")

        print(f"  {split_name:5s}: {len(split_samples):>5} samples -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build LROC-Twin instruction-tuning dataset from LUCID.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Max samples to process (default: 2000). Set to 0 for full dataset.",
    )
    args = parser.parse_args()

    print("=== LROC-Twin: Dataset Builder ===")
    print(f"Source  : {HF_DATASET_ID} / {HF_CONFIG}")
    print(f"Cap     : {args.max_samples if args.max_samples > 0 else 'full dataset'}")
    print(f"Images  : {IMAGE_DIR}")
    print(f"Output  : {OUTPUT_DIR}\n")

    samples = process_dataset(args.max_samples)

    if not samples:
        print("[ERROR] No samples processed. Check HuggingFace connectivity and dataset ID.")
        return

    print("\nWriting splits:")
    split_and_write(samples)

    total = len(samples)
    print(f"\nDataset summary:")
    print(f"  Total  : {total}")
    print(f"  Train  : ~{int(total * TRAIN_RATIO)}")
    print(f"  Val    : ~{int(total * VAL_RATIO)}")
    print(f"  Test   : ~{int(total * TEST_RATIO)}")
    print("\nDone. Run verify_dataset.py to validate splits.")


if __name__ == "__main__":
    main()
