# kNN-VC reproduction

Reproduction/study repo for [kNN-VC](https://github.com/bshall/knn-vc) (any-to-any voice
conversion via k-nearest-neighbors regression in WavLM feature space).

## Status

**Tier 1 only** — inference with pretrained checkpoints, no training.

## Layout

- `docs/notes/knn-vc.md` — paper notes.
- `docs/maths.md` — math primer for the pipeline (cosine distance, kNN averaging, mel-spectrogram).
- `docs/plan.md` — Tier 1 notebook plan.
- `docs/resource_estimate.md` — compute/data budget per tier.
- `notebooks/01_tier1_inference.ipynb` — Tier 1 pipeline, one pipeline stage per cell:
  load pretrained WavLM/HiFi-GAN, pull LibriSpeech dev-clean audio, extract WavLM layer-6
  features, hand-rolled kNN matching (cross-checked against the reference `.match()` call),
  vocode, listen.
- `notebooks/model.ipynb` — scratch/exploration notebook.

## Setup

```
pip install -r requirements.txt
jupyter notebook notebooks/01_tier1_inference.ipynb
```

LibriSpeech dev-clean (~337MB) auto-downloads into `notebooks/data/` on first run (gitignored).
Device auto-detects `cuda` → `mps` → `cpu`.
