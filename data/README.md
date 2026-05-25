# Data

All raw imagery and processed splits are excluded from version control via `.gitignore`.
Large files are stored locally during development and released to HuggingFace Hub in Phase 4.

## LUCID (instruction-tuning)

**Source**: `pcvlab/lucid` on HuggingFace  
**Config used**: `stage2_qa` (20K image-conversation pairs)  
**Used for**: Phase 2 fine-tuning and in-domain (Tier 1) evaluation  

### Schema

Each record in `stage2_qa` has:

| Field | Type | Description |
|---|---|---|
| `id` | str | Geographic coordinate identifier e.g. `N31.200_W05.500` |
| `image` | PIL Image | 224x224 grayscale lunar surface image |
| `image_path` | str | Original path within the LUCID release |
| `domain_lunar` | bool | Always True for stage2_qa |
| `domain_panchro` | bool | Panchromatic image flag |
| `conversations` | list[dict] | Multi-turn LLaVA format: `[{from: human, value: ...}, {from: gpt, value: ...}]` |

There are no structured hazard, safe zone, or terrain type fields.
Hazard understanding is learned from the geomorphology conversation content.

### Acquisition

`build_dataset.py` downloads `stage2_qa` via streaming and saves locally:

```bash
python training/scripts/build_dataset.py --max-samples 2000
```

Default cap is 2000 samples for fast iteration. Use `--max-samples 0` for the full 20K dataset before final training.

### Local layout after download

```
data/
  lucid/
    images/          PNG files saved from HuggingFace PIL images
  processed/
    lucid_train.jsonl
    lucid_val.jsonl
    lucid_test.jsonl
```

## NASA PDS (zero-shot OOD — Tier 2)

**Source**: https://pds-imaging.jpl.nasa.gov/  
**Used for**: Phase 2 zero-shot generalization evaluation  
Place downloaded files under `data/nasa_pds/`.

## Chang'e-4 (zero-shot OOD — Tier 2)

**Source**: https://moon.bao.ac.cn/  
**Used for**: Phase 2 zero-shot generalization evaluation  
Place downloaded files under `data/change4/`.

## Processed JSONL format

Each line in the JSONL splits is a JSON object:

```json
{
  "id": "N31.200_W05.500",
  "image_path": "data/lucid/images/N31_200_W05_500.png",
  "prompt": "You are a lunar surface analyst... [system prompt + human turns]",
  "response": "The mare plains have a diffuse boundary...",
  "split": "train",
  "source": "lucid"
}
```
