"""
dataset_loader.py

PyTorch Dataset for LROC-Twin instruction-tuning JSONL splits.
Handles image loading, chat template formatting for Qwen2-VL,
and tokenization.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor


class LROCDataset(Dataset):
    """
    Loads LROC-Twin JSONL split and formats each sample
    as a Qwen2-VL chat template input ready for causal LM training.
    """

    def __init__(
        self,
        jsonl_path: str,
        processor: AutoProcessor,
        max_seq_length: int = 1024,
        image_max_pixels: int = 200704,
    ):
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.image_max_pixels = image_max_pixels
        self.samples = self._load(jsonl_path)

    def _load(self, path: str) -> list[dict]:
        samples = []
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                if Path(record["image_path"]).exists():
                    samples.append(record)
        print(f"Loaded {len(samples)} samples from {path}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        record = self.samples[idx]

        image = Image.open(record["image_path"]).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": record["prompt"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": record["response"]}],
            },
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.squeeze(0)

        # Mask prompt tokens in labels so loss is computed on response only
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values

        return result
