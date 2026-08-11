# Tier 1 kNN-VC reproduction — inference-only notebook

## Context

Paper note (`docs/notes/knn-vc.md`), math primer (`docs/maths.md`), and resource estimate (`docs/resource_estimate.md`) are done. Locking onto Tier 1 (inference-only, use released pretrained checkpoints, no training) and implementing it as a Jupyter notebook where every stage of the pipeline is a separate, inspectable cell — not one black-box `.match()` call. Goal is pedagogical: see the actual math from `maths.md` (cosine distance, kNN averaging, mel-spectrogram) happen on real tensors, not just call a wrapper.

Confirmed via reading the official repo (`bshall/knn-vc`) source (`matcher.py`) that the `torch.hub.load('bshall/knn-vc', 'knn_vc', ...)` wrapper exposes reusable sub-components as plain attributes:
- `knn_vc.wavlm` — frozen WavLM-Large encoder (already configured to pull layer 6 internally via `get_features`)
- `knn_vc.hifigan` — HiFi-GAN vocoder (prematched-trained variant if `prematched=True`)
- `knn_vc.sr` = 16000, `knn_vc.hop_length` = 320 (20ms/frame, confirms `maths.md` §3 framing)
- `knn_vc.get_features(path)` — encoder step
- `knn_vc.get_matching_set(paths)` — concatenates reference features
- `knn_vc.match(query, matching_set, topk=4)` — does cosine-dist + topk + average + vocode in one call, internally via a `fast_cosine_dist` helper: `1 - (dotprod / (norm1*norm2))`, then `topk(k, largest=False)`, then `synth_set[best.indices].mean(dim=1)`

Plan: reuse `knn_vc.wavlm` and `knn_vc.hifigan` as the pretrained black boxes (per Tier 1 scope — no training, no reimplementing WavLM/HiFi-GAN), but **hand-roll the kNN matching step ourselves** in a visible cell (replicate `fast_cosine_dist` + topk + mean in ~5 lines of torch) instead of calling `.match()` directly. End by calling `.match()` once too and diffing against our manual result, as a correctness check that the "visible" version is faithful to the reference implementation.

User decisions:
- Audio: auto-download LibriSpeech dev-clean via `torchaudio.datasets.LIBRISPEECH(download=True)`, pick one source speaker utterance + a few utterances from a different speaker as the reference/matching set. Self-contained, no manual file prep, matches paper's own eval data.
- Device: auto-detect (`cuda` → `mps` → `cpu` fallback).

## Files to create

- `requirements.txt` — `torch>=2.0`, `torchaudio`, `numpy`, `matplotlib`, `jupyter` (no sklearn — use `torch.pca_lowrank` for the 2D feature-space viz, no scipy/whisper — WER/EER checks are Tier 3, out of scope here).
- `notebooks/01_tier1_inference.ipynb` — the notebook itself.
- `data/` — gitignored download target for LibriSpeech (check existing `.gitignore` covers this, extend if not).

## Notebook cell-by-cell plan

1. **Markdown intro** — states this is Tier 1 (inference-only, pretrained checkpoints, no training), links back to `docs/notes/knn-vc.md` and `docs/maths.md` sections by name so each code cell can reference "this is §1 / §3 from maths.md".
2. **Setup** — imports (`torch`, `torchaudio`, `matplotlib`, `IPython.display.Audio`), device auto-detect cell, print torch/torchaudio versions (guard: paper repo requires torch>=2.0, python>=3.10).
3. **Load pretrained pipeline** — `torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)`. Print: WavLM param count, HiFi-GAN param count, `knn_vc.sr`, `knn_vc.hop_length` — ties straight back to maths.md §2/§3.
4. **Get audio** — `torchaudio.datasets.LIBRISPEECH(root='./data', url='dev-clean', download=True)`. Pick 1 utterance from speaker A as source/query, 3-5 utterances from speaker B as reference set. Plot waveforms, play both inline (`IPython.display.Audio`).
5. **Encoder step (maths.md §2)** — call `knn_vc.get_features()` separately on source and each reference utterance (not the batched `get_matching_set` yet) so per-file shapes are visible: print `(T, 1024)` per file, confirm `T ≈ duration_sec / 0.02` (20ms hop). Visualize: `torch.pca_lowrank` down to 2D, scatter source-speaker frames vs reference-speaker frames in different colors — makes the "phonetic clustering in feature space" claim from the paper visually concrete.
6. **Matching set (maths.md §1)** — concatenate reference features into one matching-set tensor via `get_matching_set`, print total frame count.
7. **Manual kNN matching (maths.md §1, hand-rolled)** — reimplement `fast_cosine_dist` ourselves in a cell: `dist = 1 - (query @ matching_set.T) / (‖query‖ ‖matching_set‖)`, `topk(k=4, largest=False)`, average selected vectors. For 2-3 example query frame indices, print/plot which reference frame indices got picked and their distances — makes the "unit selection" mechanism concrete frame-by-frame.
8. **Vocoder step (maths.md §3/§4, black box)** — feed our manually-averaged feature sequence into `knn_vc.hifigan(...)` to synthesize the waveform. Plot mel-spectrograms of source vs reference vs converted side-by-side (`torchaudio.transforms.MelSpectrogram` matching the paper's 128-dim/10ms-hop/64ms-Hann config from maths.md §3).
9. **Listen** — play source, reference, converted audio inline for a subjective gut-check.
10. **Correctness check against reference implementation** — call `knn_vc.match(query_seq, matching_set, topk=4)` directly (the official one-liner) and confirm our manual step-by-step output matches (e.g. `torch.allclose` or max-abs-diff near zero) — proves the "visible" version isn't a divergent reimplementation.
11. **Wrap-up markdown** — what was demonstrated, explicit note that this is Tier 1 only (no WER/EER/MOS — those need Tier 2/3 per `resource_estimate.md`), pointer to Tier 2 as next step.

## Verification

- Run notebook top-to-bottom (`jupyter nbconvert --to notebook --execute` or interactively) on this machine — confirm every cell executes without error, audio downloads once and caches in `data/`, and step 10's correctness check passes (manual pipeline matches official `.match()` output).
- Sanity-listen to the converted audio output to confirm it's audibly the source content in the reference speaker's voice (matches paper's demo page qualitatively).
