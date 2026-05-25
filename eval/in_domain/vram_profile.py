"""
vram_profile.py

Measures GPU memory usage and inference latency for Qwen2-VL-2B
across quantization configurations.

Run this before any training to establish the VRAM budget baseline
and verify the hardware constraint is respected.

Output: eval/in_domain/results/vram_profile.json
"""

import json
import time
import torch
from pathlib import Path
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np
import tempfile

OUTPUT_DIR = Path("eval/in_domain/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DUMMY_IMAGE_SIZE = (336, 336)
DUMMY_PROMPT = "Analyze this lunar terrain image for hazards."
N_WARMUP = 2
N_MEASURE = 5

CONFIGS = [
    {"name": "fp16",       "load_in_4bit": False, "load_in_8bit": False, "torch_dtype": torch.float16},
    {"name": "int8",       "load_in_4bit": False, "load_in_8bit": True,  "torch_dtype": torch.float16},
    {"name": "int4_bnb",   "load_in_4bit": True,  "load_in_8bit": False, "torch_dtype": torch.float16},
]


def make_dummy_image() -> str:
    img = Image.fromarray(
        np.random.randint(0, 128, (*DUMMY_IMAGE_SIZE, 3), dtype=np.uint8)
    )
    path = Path(tempfile.gettempdir()) / "lroc_dummy.png"
    img.save(path)
    return str(path)


def measure_config(config: dict, dummy_image_path: str) -> dict:
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    load_kwargs = {
        "device_map": "auto",
        "torch_dtype": config["torch_dtype"],
    }
    if config["load_in_4bit"]:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif config["load_in_8bit"]:
        load_kwargs["load_in_8bit"] = True

    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, **load_kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    mem_after_load = torch.cuda.memory_allocated() / 1e9

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": dummy_image_path},
                {"type": "text",  "text": DUMMY_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to("cuda")

    for _ in range(N_WARMUP):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=64)

    latencies = []
    for _ in range(N_MEASURE):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=64)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    del model
    torch.cuda.empty_cache()

    return {
        "config": config["name"],
        "vram_after_load_gb": round(mem_after_load, 3),
        "vram_peak_gb": round(peak_mem, 3),
        "fits_4gb_budget": peak_mem < 4.0,
        "latency_mean_s": round(sum(latencies) / len(latencies), 3),
        "latency_min_s": round(min(latencies), 3),
        "latency_max_s": round(max(latencies), 3),
    }


def main():
    print("=== LROC-Twin: VRAM Profiler ===\n")
    dummy_image = make_dummy_image()
    results = []

    for config in CONFIGS:
        print(f"Measuring {config['name']}...")
        result = measure_config(config, dummy_image)
        results.append(result)
        print(f"  Peak VRAM : {result.get('vram_peak_gb', 'N/A')} GB")
        print(f"  Fits 4GB  : {result.get('fits_4gb_budget', 'N/A')}")
        print(f"  Latency   : {result.get('latency_mean_s', 'N/A')}s\n")

    output_path = OUTPUT_DIR / "vram_profile.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Profile saved to {output_path}")

    print("\n=== Summary ===")
    for r in results:
        fits = r.get("fits_4gb_budget", False)
        status = "PASS" if fits else "FAIL"
        print(f"  [{status}] {r.get('config', '?'):12s} — "
              f"{r.get('vram_peak_gb', '?')} GB peak, "
              f"{r.get('latency_mean_s', '?')}s latency")


if __name__ == "__main__":
    main()
