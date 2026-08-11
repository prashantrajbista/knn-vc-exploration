# kNN-VC reproduction

Reproduction/study repo for [kNN-VC](https://github.com/bshall/knn-vc) (any-to-any voice
conversion via k-nearest-neighbors regression in WavLM feature space).

## Status

**Tier 1** — inference with pretrained checkpoints, no training. Done.
**Tier 2** — objective WER/CER/EER reproduction of Table 1, pretrained checkpoints only, no training. Done — see [Results](#results-tier-2-vs-paper) below.

## Layout

- `docs/notes/knn-vc.md` — paper notes.
- `docs/maths.md` — math primer for the pipeline (cosine distance, kNN averaging, mel-spectrogram).
- `docs/plan.md` — Tier 1 notebook plan.
- `docs/plan_tier_2.md` — Tier 2 eval plan (WER/CER/EER on LibriSpeech test-clean).
- `docs/resource_estimate.md` — compute/data budget per tier.
- `notebooks/01_tier1_inference.ipynb` — Tier 1 pipeline, one pipeline stage per cell:
  load pretrained WavLM/HiFi-GAN, pull LibriSpeech dev-clean audio, extract WavLM layer-6
  features, hand-rolled kNN matching (cross-checked against the reference `.match()` call),
  vocode, listen.
- `notebooks/model.ipynb` — scratch/exploration notebook.
- `scripts/tier2_eval.py` — Tier 2 batch eval: converts 200 LibriSpeech test-clean utterances
  (5 per speaker x 40 speakers) to each of the 39 other speakers (7800 outputs), scores WER/CER
  with Whisper-base and EER with a pretrained speaker embedding model. Writes
  `data/tier2_outputs/results.json` + `tier2.log`. `--sanity` for a 2-conversion smoke test,
  `--stage convert|asr|eer` to run one stage at a time (conversion skips wavs that already exist,
  so it's resumable).

## Setup

```
pip install -r requirements.txt
jupyter notebook notebooks/01_tier1_inference.ipynb
```

LibriSpeech dev-clean (~337MB) auto-downloads into `notebooks/data/` on first run (gitignored).
Device auto-detects `cuda` → `mps` → `cpu`.

For Tier 2:

```
python scripts/tier2_eval.py --sanity   # 2-conversion smoke test first
python scripts/tier2_eval.py            # full run: 40 speakers x 5 utterances x 39 targets
```

Downloads LibriSpeech test-clean (~350MB) into `notebooks/data/`. GPU strongly recommended —
7800 conversions + ASR + speaker-embedding passes is slow on CPU.

## Results (Tier 2 vs. paper)

Full run (200 source utterances x 39 targets = 7800 conversions, `n_eer_scores` = 15600),
against Table 1 of the paper (LibriSpeech test-clean, kNN-VC row and topline row):

| Metric              | Ours   | Paper | Delta  |
|----------------------|--------|-------|--------|
| WER (converted)       | 7.81   | 7.36  | +0.45  |
| CER (converted)       | 3.54   | 2.96  | +0.58  |
| WER (topline, real)   | 6.08   | 5.96  | +0.12  |
| CER (topline, real)   | 2.67   | 2.38  | +0.29  |
| EER (%, higher=better)| 35.99  | 37.15 | -1.16  |

Raw output: `data/tier2_outputs/results.json` (gitignored, regenerate by running the script).

### Why the numbers don't match exactly

- **Different Whisper checkpoint/decoding defaults.** Paper used Whisper-base with 2023-era
  defaults; ours pulls the current `openai-whisper` package. Small WER/CER drift is expected —
  the topline row (real, unconverted speech) already shows ~0.1-0.3 point drift before kNN-VC
  is even in the picture, which calibrates how much of the converted-row gap is ASR-version
  noise vs. actual conversion-quality difference.
- **Different eval-utterance sample.** The paper doesn't publish which 200 test-clean utterances
  (5/speaker x 40 speakers) or random seed it used. `scripts/tier2_eval.py` picks a *deterministic*
  sample (sorted by speaker then utterance ID) so our own runs are reproducible, but it isn't
  utterance-for-utterance identical to the paper's.
- **Different speaker-verification model for EER.** Paper's EER protocol (Section 4.3) traces to
  van Niekerk et al. 2022 (same authors) using an x-vector architecture (Snyder et al. 2018). The
  exact trained checkpoint isn't published alongside the paper. We use
  [`RF5/simple-speaker-embedding`](https://github.com/RF5/simple-speaker-embedding) — a GE2E
  speaker-embedding model released by kNN-VC's own first author, trained on
  VCTK+LibriSpeech+VoxCeleb1+2, self-reported LibriSpeech test-clean EER 2.95% as a verifier —
  the closest available proxy, but not confirmed to be byte-identical to whatever produced
  Table 1. (We first tried a generic SpeechBrain VoxCeleb x-vector, which gave EER ~46% — a much
  larger gap — swapping to this domain-matched model closed most of it.)
- **Enrollment-utterance sampling for EER.** Our EER computation uses one fixed enrollment
  utterance per target speaker (reused across all trials into that speaker); the paper's text
  doesn't fully specify whether it resamples a fresh enrollment utterance per trial. Likely a
  minor contributor to the residual EER gap.
- **Loudness normalization is on** (`knn_vc.match`'s default `tgt_loudness_db=-16`), matching the
  library's own default — an earlier version of this script disabled it by mistake (copied from
  a Tier 1 notebook cell that had a different reason to disable it), which alone inflated EER by
  ~10 points before being fixed.

Given all of that, WER/CER land within ~0.5 points and EER within ~1.2 points of the paper using
only pretrained, off-the-shelf checkpoints and no training — close enough to call the headline
claim (kNN-VC roughly matches strong baselines on intelligibility while far exceeding them on
speaker similarity) reproduced, without claiming exact digit-for-digit replication.
