# LROC-Twin

Domain-specific vision-language model distillation for autonomous lunar navigation.

Fine-tunes Qwen2-VL-2B on the LUCID lunar geomorphology dataset using 4-bit QLoRA within a 4GB VRAM constraint, then embeds the model in a LangGraph navigation agent evaluated in NVIDIA Isaac Sim.

## Results

| Model | LUCID F1 | NASA PDS F1 | Chang'e-4 F1 | MES |
|---|---|---|---|---|
| Qwen2-VL-2B (frozen) | - | - | - | - |
| LROC-Twin (LoRA r=8) | - | - | - | - |
| LROC-Twin (LoRA r=16) | - | - | - | - |
| LROC-Twin (LoRA r=32) | - | - | - | - |

*Results populated after Phase 2 evaluation.*

## Architecture

- **Vision backbone**: Qwen2-VL-2B with Unsloth QLoRA (4-bit)
- **Training data**: LUCID dataset (open-access lunar geomorphology)
- **Evaluation**: 3-tier protocol — in-domain, zero-shot OOD, live agent
- **Agent**: LangGraph ReAct loop inside NVIDIA Isaac Sim
- **Primary metric**: Mission Efficiency Score = Completion Rate × (1 − Collision Frequency)

## Quickstart

*Instructions added after Phase 4 packaging.*

## Project structure

```
data/       LUCID, NASA PDS, Chang'e-4 imagery and processed splits
training/   QLoRA fine-tuning configs and scripts
eval/       Evaluation harnesses for all three tiers
agent/      LangGraph perception-reasoning-action agent
sim/        NVIDIA Isaac Sim environment and scenario scripts
demo/       Gradio demo application
```

## Hardware requirements

Minimum: 4GB VRAM GPU (tested on RTX 3050 class). Training uses Unsloth with 4-bit QLoRA and gradient checkpointing.

## License

MIT
