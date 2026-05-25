# Evaluation

Three-tier evaluation protocol for LROC-Twin.

## Tier 1: in-domain (LUCID test split)

**Script**: `eval/in_domain/baseline.py`
**Metric**: Hazard F1 (macro), precision, recall
**Purpose**: Measures fine-tuning gain over frozen baseline on held-out LUCID samples.

Run baseline first, record metrics, then run the same script against the fine-tuned adapter in Phase 2.

## Tier 2: zero-shot OOD

**Scripts**: `eval/ood/` (populated in Phase 2)
**Datasets**: NASA PDS imagery, Chang'e-4 imagery
**Metric**: Hazard F1 on unseen domain
**Purpose**: Tests whether fine-tuning generalizes beyond LUCID distribution.

## Tier 3: live agent

**Scripts**: `eval/agent/` (populated in Phase 3)
**Environment**: NVIDIA Isaac Sim
**Metric**: Mission Efficiency Score = Completion Rate × (1 − Collision Frequency)
**Purpose**: End-to-end navigation performance in a physics-simulated lunar environment.

## VRAM profiling

**Script**: `eval/in_domain/vram_profile.py`
**Purpose**: Verifies the 4GB VRAM budget constraint across quantization configs before training.
Run this first on any new hardware before committing to a training run.
