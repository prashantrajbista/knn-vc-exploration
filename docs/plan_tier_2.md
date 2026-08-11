# Tier 2 kNN-VC reproduction — objective WER/CER/EER on LibriSpeech test-clean

## Context

Tier 1 (`docs/plan.md`, `notebooks/01_tier1_inference.ipynb`) already loads the **pretrained, prematched** HiFi-GAN checkpoint (`torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, pretrained=True)`) plus frozen WavLM-Large. That means the vocoder-training assumption baked into `docs/resource_estimate.md`'s original "Tier 2" (train HiFi-GAN yourself, 3-5 days/variant, ~$150-250) is no longer the blocker — the paper's own released checkpoint already is the prematched variant Table 1 reports on. **This plan supersedes that entry**: no training anywhere in this tier.

Goal: reproduce the **objective** row of Table 1 (kNN-VC: WER 7.36, CER 2.96, EER 37.15%) and the topline row (WER 5.96, CER 2.38) — using only pretrained, off-the-shelf models, following the paper's §4.3 protocol as closely as practical. Subjective MOS/SIM (Mechanical Turk) is out of scope — that's a further tier if ever pursued (see `docs/resource_estimate.md` Tier 3 note on cost/turnaround).

Paper protocol being targeted (§4.3, confirmed by reading the PDF directly):
- 200 source utterances sampled from LibriSpeech **test-clean** (5 per speaker × 40 speakers).
- Each source utterance converted to the **39 other** speakers → 7800 converted outputs.
- **WER/CER**: Whisper-base ASR, default decoding params, transcribe converted speech, compare to ground-truth transcript.
- **EER**: pretrained x-vector speaker-verification system, cosine similarity between x-vectors. Score set = converted-vs-target-enrollment pairs (label 0) + genuine target-vs-different-target-enrollment pairs (label 1), combined, EER computed over both. Max possible EER = 50% (indistinguishable from genuine).

## Files to create

- `docs/plan_tier_2.md` — this file.
- `scripts/tier2_eval.py` — batch job, not a notebook. 7800 conversions + ASR + speaker-verification passes will run for hours; needs to be resumable/checkpointed, which is awkward in a notebook. Writes intermediate state (converted wavs, transcripts, scores) to disk incrementally so a crash/interrupt doesn't lose progress.
- `notebooks/03_tier2_analysis.ipynb` — thin notebook that loads `scripts/tier2_eval.py`'s output (a scores CSV/JSON) and reports final WER/CER/EER numbers next to the paper's Table 1, for a readable side-by-side.
- `data/test-clean/` — gitignored, auto-downloaded LibriSpeech test-clean (~350MB).
- `data/tier2_outputs/` — gitignored, converted wavs + transcripts + score arrays.
- `requirements.txt` additions: `jiwer` (WER/CER), `scikit-learn` (EER via `roc_curve`), `openai-whisper` or `transformers` (Whisper-base), `speechbrain` (pretrained x-vector).

## Step-by-step plan

1. **Data prep.** Download `test-clean` via `torchaudio.datasets.LIBRISPEECH(root='./data', url='test-clean', download=True)`. Select 40 speakers × 5 utterances deterministically (sort by speaker ID then utterance ID, take first 5) — paper doesn't publish its exact sample or a seed, so exact utterance-for-utterance match isn't possible; deterministic selection makes *our* run reproducible even if it isn't *identical* to the paper's.

2. **Matching sets per target speaker.** For each of the 40 speakers, build the WavLM matching set from all their test-clean audio **excluding** the 5 utterances reserved as eval sources for that speaker (avoid leaking an eval utterance into its own target's matching set). This mirrors the paper's "~8 minutes per speaker" setup.

3. **Batch conversion.** For each of the 200 source utterances, convert to each of the 39 other speakers using the existing pretrained pipeline (`knn_vc.get_features` → hand-rolled or `.match()` kNN → `knn_vc.hifigan`, reusing Tier 1 code) = 7800 conversions. Write each output wav to `data/tier2_outputs/wavs/`, checkpoint progress (e.g. a manifest CSV of what's done) so the script can resume after interruption.

4. **ASR pass (WER/CER).** Transcribe all 7800 converted wavs with Whisper-base, default decoding. Compare against each source utterance's ground-truth LibriSpeech transcript using `jiwer`. Also run Whisper-base on the 200 **original** (unconverted) source utterances to get our own topline WER/CER, for comparison against the paper's 5.96/2.38 — this cross-checks that our ASR setup is calibrated similarly to the paper's before trusting the kNN-VC numbers.

5. **Speaker-verification pass (EER).** Embed all 7800 converted wavs with a pretrained x-vector model (SpeechBrain `speechbrain/spkrec-xvect-voxceleb`, closest off-the-shelf match to the paper's ref [23]). For each converted sample, pick one held-out genuine enrollment utterance of the *target* speaker (not used as matching-set or source) and compute cosine similarity → "converted" score (label 0). Separately, sample an equal number of genuine same-speaker pairs (two different held-out utterances of the same speaker) → "genuine" score (label 1). Combine both score sets, compute EER via `sklearn.metrics.roc_curve` (EER = point where FPR = FNR).

6. **Sanity subset before full run.** Full 7800-conversion + ASR + EER run is hours long — first run the whole pipeline on a tiny slice (e.g. 2 speakers × 2 source utterances × all 39 targets = 156 conversions) to catch bugs cheaply before committing to the full run.

7. **Aggregate + report.** `notebooks/03_tier2_analysis.ipynb` loads the manifest + scores, prints our WER/CER/EER + topline WER/CER next to Table 1's numbers.

## Compute / storage / time estimate

- 7800 conversions: encoder + kNN + vocoder inference only (no training) — same per-utterance cost as Tier 1, scaled up. Rough budget: a few hours on `mps`/`cuda`, longer on CPU.
- Whisper-base transcription (~8000 short clips): tens of minutes on GPU, longer on CPU/`mps`.
- x-vector embeddings (~8000+ clips): fast, minutes.
- Storage: ~8000 short 16kHz wavs, likely 1-3GB total, gitignored.
- Cost: ~$0 on local GPU/`mps`; low single-digit $ on rented GPU if needed.

## Verification

- Compare our WER/CER/EER against Table 1's kNN-VC row (7.36 / 2.96 / 37.15%) and topline row (5.96 / 2.38). Treat close-but-not-identical as success — the paper's exact Whisper checkpoint version, exact x-vector model, and exact utterance sample aren't published, so exact digit-for-digit match isn't the bar. A topline WER within ~1 point of 5.96 is the calibration check that our ASR setup is comparable; large topline divergence means fix the ASR setup before trusting the kNN-VC numbers.
- Sanity subset (step 6) passes end-to-end without shape/device errors before launching the full 7800-run.

## Open questions to resolve while implementing

- Whether the public `bshall/knn-vc` repo ships its own eval script — reuse it if so, rather than re-deriving x-vector model choice and EER math from scratch.
- Exact Whisper "default decoding params" and text normalization the paper used (case/punctuation stripping before WER) — `jiwer`'s default normalization is a reasonable stand-in, but flag this as a source of small WER deltas vs. the paper.
- Whether to also implement the **plain** (non-prematched) HiFi-GAN checkpoint as a secondary run, since Figure 2's ablation claim (prematching improves EER/WER at every data size) would need it — out of scope for matching Table 1, but cheap to add later since it's also a pretrained checkpoint (`prematched=False`), no training either way.
