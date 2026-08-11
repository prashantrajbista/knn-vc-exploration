# Tier 3 — kNN-VC as a speaker anonymizer, evaluated under VoicePrivacy Challenge 2024

## Context

Tiers 1-2 reproduce kNN-VC as a *voice conversion* system (convert to a specific target speaker,
measure intelligibility + speaker similarity). This tier asks a different question, flagged as an
open gap in the paper notes (`docs/notes/knn-vc.md`, Inferences section): the paper never tests
kNN-VC as an **anonymization** technique — it measures conversion quality/similarity, not
adversarial re-identification risk. This tier wires kNN-VC into the
[VoicePrivacy Challenge 2024](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024)
(VPC2024) evaluation toolkit to answer that.

**Scope decision (confirmed with user):** LibriSpeech-only. VPC2024's IEMOCAP-based emotion/UAR
utility metric is out of scope — IEMOCAP requires a manual USC license request with a ~7-9 day
approval wait, and privacy (ASV attacker EER) + WER utility is the core question anyway. This can
be revisited as a later phase if IEMOCAP access is requested separately.

Two structural facts about VPC2024 that shape this whole plan:

1. **It's a black-box wav-in, metrics-out pipeline, not a library you call into.** There's no
   documented plugin interface for a custom anonymization system. You produce anonymized `.wav`
   files in a fixed 9-folder layout, then run VPC's own `run_evaluation.py`, which handles ASV
   attacker retraining + WER internally. `run_anonymization.py` and all five built-in baselines
   (McAdams, STTTS, NAC, ASR-BN+VQ, x-vectors) are irrelevant to us — we bypass that script
   entirely and only ever produce the folders it would have produced.
2. **kNN-VC has no pseudo-speaker assignment logic today, and needs one.** The paper's kNN-VC
   always converts to a fixed real target speaker. For anonymization that's meaningless — it
   doesn't remove identity, it just swaps the source speaker's real identity for the target
   speaker's real identity (whoever attacks the output re-identifies the *target*, defeating the
   point). VPC's own baselines solve this by pooling/blending multiple real speakers so the output
   maps to no single real identity. kNN-VC's matching-set mechanism (a pool of reference frames,
   k-NN averaged per query frame) is naturally suited to the same trick: build the matching set
   from **many speakers**, not one, so each output frame is a blend across identities. This is new
   design work — nothing to copy from the paper or from `scripts/tier2_eval.py`'s single-target
   conversion.

## Required output structure (mandatory folders only, IEMOCAP excluded)

```
data/libri_dev_enrolls${suffix}/wav/*.wav
data/libri_dev_trials_m${suffix}/wav/*.wav
data/libri_dev_trials_f${suffix}/wav/*.wav
data/libri_test_enrolls${suffix}/wav/*.wav
data/libri_test_trials_m${suffix}/wav/*.wav
data/libri_test_trials_f${suffix}/wav/*.wav
```

`${suffix}` is an arbitrary tag for this run (e.g. `_knnvc`) — VPC's eval script diffs the
original vs. `${suffix}`-tagged folders to compute privacy/utility deltas.

**Resolved (was an open question):** confirmed directly from `eval_post.yaml` +
`evaluation/privacy/asv/asv_train/hparams/train_ecapa_tdnn_small.yaml` in the actual toolkit
source — `train-clean-360` anonymization is mandatory **only** for the semi-informed (ranking)
attacker, which retrains an ECAPA-TDNN ASV model from scratch (10 epochs, 921 speakers) on
`data/train-clean-360${suffix}`. The ignorant attacker (`eval_pre.yaml`) uses a fixed pretrained
`asv_orig` model with no retraining and needs only the 6 folders above. This splits the tier into
two phases:

- **Phase A** (this implementation): ignorant-attacker privacy EER + WER, LibriSpeech dev/test
  only, ~4255 short utterances — comparable in scale to Tier 2.
- **Phase B** (later, not yet built): semi-informed/ranking attacker — additionally anonymize all
  of `train-clean-360` (360 hours, public via OpenSLR, no password) and retrain ECAPA-TDNN on it.
  Much bigger compute (the actual number comparable to the published VPC2024 leaderboard);
  deliberately deferred.

**Also resolved: the password gate is avoidable entirely, for Phase A.** VPC2024's
`libri_dev`/`libri_test` raw audio download is gated behind a registration password (SFTP from
`voiceprivacychallenge.univ-avignon.fr`). But VPC's `data.zip` (public, no password, a GitHub
release asset) ships the *metadata* for their official enroll/trial partition — `wav.scp`,
`utt2spk`, `trials`, `spk2gender` — referencing standard LibriSpeech utterance IDs (e.g.
`1272-128104-0000`). Checked directly: every one of the 2321 dev-partition and 1934
test-partition utterance IDs `data.zip` needs resolves against the `dev-clean`/`test-clean`
already downloaded in this repo from Tiers 1-2 (100% coverage, 0 missing). So the exact official
VPC partition can be reconstructed from data already on hand — no registration, no wait, and
still the real official partition (not an approximation). `train-clean-360` (Phase B) is separately
public via OpenSLR directly, also no password.

**Also resolved: `00_install.sh` is Linux+CUDA-only** (hardcoded `micromamba-linux-64` binary,
pinned CUDA 11.7 / torch 2.0.1 in an isolated micromamba env). Won't run on this Mac — the
toolkit install + `run_evaluation.py` must run on the same cloud GPU box used for Tier 2. Data
reconstruction and anonymization can be developed/sanity-tested locally on CPU first (both are
plain PyTorch, no toolkit-specific dependency), then the anonymized output handed to the cloud
box's toolkit checkout for the actual evaluation run.

## Files

- `docs/plan_tier_3.md` — this file.
- `third_party/Voice-Privacy-Challenge-2024/` — the toolkit, cloned per its own `00_install.sh` /
  `env.sh` (gitignored — external toolkit, not our code). Install must happen on a Linux+CUDA box
  (the cloud GPU box from Tier 2), not this Mac.
- `scripts/tier3_prepare_data.py` — **done, sanity-tested locally.** Downloads VPC's public
  `data.zip` metadata and reconstructs `data/libri_{dev,test}/wav/<uttid>/<uttid>.wav` from the
  `dev-clean`/`test-clean` already in this repo. Verified 100% utterance-ID resolution
  (2321/2321 dev, 1934/1934 test).
- `scripts/tier3_pseudo_speaker.py` — **done.** Pure assignment logic (no torch), separately
  testable. Utterance-level, deterministic: each utterance's donor pool = 8 donor speakers
  (`N_DONORS`), sampled via `random.Random(sha256(utt_id))` from all speakers in the raw pool
  excluding the utterance's own real speaker, 3 utterances per donor (`UTTS_PER_DONOR`) as the
  matching-set material.
- `scripts/tier3_anonymize.py` — **done, sanity-tested locally (5-utterance smoke test, CPU).**
  Reuses the encode → kNN-match → vocode pipeline proven in `scripts/tier2_eval.py` (loudness
  normalization on, per the Tier 2 loudness-bug lesson), matching set built from
  `tier3_pseudo_speaker`'s donor pool instead of one fixed real target. Caches donor WavLM
  features across utterances (donors repeat a lot across ~4255 queries drawing 24 donor
  utterances each). Writes the 6 required `${suffix}` folders.
- `configs/tier3/eval_pre_librispeech_only.yaml` — **done.** Copy of the toolkit's own
  `eval_pre.yaml` with `IEMOCAP_dev`/`IEMOCAP_test` and the `ser` step removed — Phase A only
  needs `privacy.asv` (ignorant attacker) + `utility.asr` (WER) on `libri_dev`/`libri_test`.

## Step-by-step plan

1. ~~Install the toolkit, resolve `train-clean-360` question~~ — done, see above.
2. ~~Design the pseudo-speaker assignment scheme~~ — done, see `tier3_pseudo_speaker.py` above.
   N=8 donors / 3 utterances each was picked as a reasonable starting point, not tuned — flagged
   below as worth a small ablation later, not blocking Phase A.
3. ~~Anonymize the 6 mandatory folders~~ — script done and sanity-tested locally; **the full
   4255-utterance run itself still needs to happen on the cloud GPU box** (CPU-only sanity test
   confirmed the pipeline works, but full-scale run needs GPU like Tier 2 did).
4. **Install the toolkit on the cloud box**, run `scripts/tier3_prepare_data.py` there too (or
   copy the reconstructed `data/libri_dev`, `data/libri_test` pools over), run
   `scripts/tier3_anonymize.py` for the full 4255 utterances, then run VPC's own
   `run_evaluation.py --config configs/tier3/eval_pre_librispeech_only.yaml` unmodified — this is
   where the ignorant-attacker EER and WER numbers come from; no custom eval code needed here,
   unlike Tiers 1-2 where the eval itself was built from scratch.
5. **Report**: ignorant-attacker EER and WER, and how kNN-VC-as-anonymizer compares to VPC2024's
   own published baseline numbers (B2-B6, in `results/` in the toolkit repo) — with the caveat
   that those are semi-informed-attacker numbers (Phase B), so only roughly comparable until
   Phase B exists.

## Verification

- Toolkit installs cleanly and its own baseline (e.g. B2/McAdams, CPU-only, fastest) runs
  end-to-end first, as a smoke test that the evaluation harness itself works before pointing it at
  kNN-VC output — isolates "our anonymization is wrong" from "we set up VPC2024 wrong."
- Pseudo-speaker assignment scheme is deterministic and documented (same source speaker always
  maps to the same donor pool across a run) so results are reproducible.
- Compare against VPC2024's own published baseline privacy/utility numbers as a sanity check that
  kNN-VC's semi-informed-attacker EER and WER land in a plausible range, not just internally
  self-consistent.

## Open questions still remaining

- **Phase B compute budget** — anonymizing `train-clean-360` (360 hours) plus an ECAPA-TDNN
  retrain (10 epochs, 921 speakers) is the single biggest unknown left, plausibly far larger than
  Tiers 1-2 combined. Not needed for Phase A.
- **N=8 donors / 3 utterances each was picked, not tuned.** Worth a small ablation later (vary N,
  check EER vs. WER tradeoff) once Phase A numbers exist as a baseline to compare against — not
  blocking, since some fixed reasonable choice was needed to build anything at all.
- Whether VPC2024 has published numbers for a kNN-VC-like baseline already (STTTS/B3 uses a
  not-dissimilar phone-sequence + speaker-embedding approach) that could sanity-check expected
  EER/WER ranges — worth checking `results/` in the toolkit repo before or after the Phase A run.
- Gender-mixing in donor pools: current assignment draws donors regardless of gender. Untested
  whether this measurably hurts naturalness/utility vs. a gender-matched donor pool — another
  candidate ablation once Phase A has a working baseline.
