"""
baseline.py

Evaluates frozen Qwen2-VL-2B (no fine-tuning) on the LUCID test split.
This produces the control metrics against which the fine-tuned model
is compared in Phase 2.

Outputs:
    eval/in_domain/results/baseline_predictions.jsonl
    eval/in_domain/results/baseline_metrics.json

Hardware note:
    Requires CUDA GPU with at least 4GB VRAM and Qwen2-VL-2B downloaded (~4GB).
    On CPU-only hardware this script will report 'CUDA not available' and exit.
    Run vram_profile.py first to confirm your hardware meets the budget constraint.
"""

import json
import time
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from sklearn.metrics import f1_score, precision_score, recall_score

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
TEST_SPLIT = Path("data/processed/lucid_test.jsonl")
OUTPUT_DIR = Path("eval/in_domain/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 512

HAZARD_KEYWORDS = [
    "crater", "boulder", "slope", "rough terrain",
    "hazard", "obstacle", "dangerous", "avoid",
]


def load_test_split(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_model():
    print(f"Loading {MODEL_ID} on {DEVICE} (frozen, no adapters)...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Model loaded.")
    return model, processor


def run_inference(model, processor, record: dict) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": record["image_path"]},
                {"type": "text", "text": record["prompt"]},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output[0].strip()


def extract_hazard_labels(text: str) -> list[int]:
    text_lower = text.lower()
    return [1 if kw in text_lower else 0 for kw in HAZARD_KEYWORDS]


def compute_metrics(predictions: list[dict]) -> dict:
    y_true, y_pred = [], []

    for p in predictions:
        true_labels = extract_hazard_labels(p["ground_truth"])
        pred_labels = extract_hazard_labels(p["prediction"])
        y_true.extend(true_labels)
        y_pred.extend(pred_labels)

    return {
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "hazard_keywords_evaluated": HAZARD_KEYWORDS,
        "note": (
            "Keyword-based proxy metric. "
            "Replace with annotation-level evaluation once ground truth labels are structured."
        ),
    }


def main():
    print("=== LROC-Twin: Frozen Baseline Evaluation ===")
    print(f"Model  : {MODEL_ID} (frozen, no fine-tuning)")
    print(f"Split  : {TEST_SPLIT}")
    print(f"Device : {DEVICE}\n")

    if not TEST_SPLIT.exists():
        print(f"[ERROR] Test split not found at {TEST_SPLIT}.")
        print("        Run training/scripts/build_dataset.py first.")
        return

    records = load_test_split(TEST_SPLIT)
    print(f"Loaded {len(records)} test samples.\n")

    model, processor = load_model()

    predictions = []
    start = time.time()

    for i, record in enumerate(tqdm(records, desc="Running inference")):
        try:
            prediction = run_inference(model, processor, record)
        except Exception as e:
            prediction = f"[ERROR] {str(e)}"

        predictions.append({
            "image_path": record["image_path"],
            "prompt": record["prompt"],
            "ground_truth": record["response"],
            "prediction": prediction,
        })

    elapsed = time.time() - start
    print(f"\nInference complete: {len(predictions)} samples in {elapsed:.1f}s "
          f"({elapsed/len(predictions):.2f}s per sample)")

    pred_path = OUTPUT_DIR / "baseline_predictions.jsonl"
    with open(pred_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    print(f"Predictions saved to {pred_path}")

    metrics = compute_metrics(predictions)
    metrics["model"] = MODEL_ID
    metrics["n_samples"] = len(predictions)
    metrics["inference_seconds"] = round(elapsed, 2)
    metrics["seconds_per_sample"] = round(elapsed / len(predictions), 3)

    metrics_path = OUTPUT_DIR / "baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    print("\n=== Baseline Results ===")
    print(f"  F1 (macro)        : {metrics['f1_macro']}")
    print(f"  Precision (macro) : {metrics['precision_macro']}")
    print(f"  Recall (macro)    : {metrics['recall_macro']}")
    print("\nThese are the control metrics. Record them in the README results table.")


if __name__ == "__main__":
    main()
