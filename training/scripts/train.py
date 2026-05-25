"""
train.py

Fine-tunes Qwen2-VL-2B using Unsloth + 4-bit QLoRA on the LUCID
instruction-tuning dataset.

Usage:
    python training/scripts/train.py --config training/configs/lora_r8.yaml
    python training/scripts/train.py --config training/configs/lora_r16.yaml
    python training/scripts/train.py --config training/configs/lora_r32.yaml

Requirements:
    CUDA GPU with at least 4GB VRAM.
    WSL2 or Linux (Unsloth CUDA kernels do not compile on native Windows).
    Run: pip install unsloth
"""

import argparse
import os
from pathlib import Path

import torch
import wandb
import yaml
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastVisionModel

from dataset_loader import LROCDataset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict):
    lora_cfg = cfg["lora"]
    quant_cfg = cfg["quantization"]

    model, processor = FastVisionModel.from_pretrained(
        cfg["base_model"],
        load_in_4bit=quant_cfg["load_in_4bit"],
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        random_state=42,
    )

    return model, processor


def build_training_args(cfg: dict) -> TrainingArguments:
    t = cfg["training"]
    output_dir = Path(t["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        fp16=t["fp16"],
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        dataloader_num_workers=t["dataloader_num_workers"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        evaluation_strategy=t["evaluation_strategy"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        report_to=t.get("report_to", "none"),
        run_name=cfg.get("run_name", "lroc-twin"),
        remove_unused_columns=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML training config (e.g. training/configs/lora_r8.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = cfg.get("run_name", "lroc-twin")

    print(f"=== LROC-Twin Training: {run_name} ===")
    print(f"Base model : {cfg['base_model']}")
    print(f"LoRA rank  : {cfg['lora']['r']}")
    print(f"4-bit QLoRA: {cfg['quantization']['load_in_4bit']}")
    print(f"Output dir : {cfg['training']['output_dir']}\n")

    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available. Training requires a CUDA GPU.")
        print("        On Windows, ensure you are running inside WSL2.")
        return

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"VRAM       : {vram_gb:.1f} GB\n")

    if cfg.get("training", {}).get("report_to") == "wandb":
        wandb.init(project="lroc-twin", name=run_name)

    model, processor = build_model(cfg)

    data_cfg = cfg["data"]
    train_dataset = LROCDataset(
        data_cfg["train_split"],
        processor,
        max_seq_length=data_cfg["max_seq_length"],
        image_max_pixels=data_cfg["image_max_pixels"],
    )
    val_dataset = LROCDataset(
        data_cfg["val_split"],
        processor,
        max_seq_length=data_cfg["max_seq_length"],
        image_max_pixels=data_cfg["image_max_pixels"],
    )

    training_args = build_training_args(cfg)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.tokenizer,
        dataset_text_field=None,
        max_seq_length=data_cfg["max_seq_length"],
    )

    print("Starting training...\n")
    trainer.train()

    adapter_path = Path(cfg["training"]["output_dir"]) / "final_adapter"
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    print(f"\nAdapter saved to {adapter_path}")

    if cfg.get("training", {}).get("report_to") == "wandb":
        wandb.finish()

    print(f"\n=== Training complete: {run_name} ===")


if __name__ == "__main__":
    main()
