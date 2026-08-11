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

**Open question to resolve before committing to a compute budget** (README text was ambiguous on
this, needs confirming directly against `configs/` or `run_evaluation.py` in the actual repo,
not just its docs): whether `train-clean-360` anonymization is *also* mandatory. The
semi-informed attacker (the metric that ranks challenge submissions) retrains its own ASV system
on anonymized data — a handful of enrollment/trial utterances likely isn't enough training data
for that, so `train-clean-360` (360 hours) is the probable attacker-training corpus. If so,
anonymizing it with kNN-VC (WavLM encode + kNN + HiFi-GAN vocode over 360 hours) is a
*much* bigger compute job than Tier 2's 7800 short clips — plausibly 50-100+ hours of wall time on
a single consumer GPU. This needs to be nailed down (by installing the toolkit and reading
`run_evaluation.py`/`configs/` directly) before scheduling this tier, not assumed away.

## Files to create

- `docs/plan_tier_3.md` — this file.
- `third_party/Voice-Privacy-Challenge-2024/` — the toolkit itself, cloned in per its own
  `00_install.sh` / `env.sh` (gitignored — this is an external toolkit, not our code).
- `scripts/tier3_pseudo_speaker.py` — the new piece. Builds a per-source-speaker (or per-utterance)
  *pooled* matching set from multiple donor speakers instead of one real target, producing the
  pseudo-speaker identity kNN-VC converts each source utterance into.
- `scripts/tier3_anonymize.py` — reuses the encoder/kNN/vocoder machinery already proven in
  `scripts/tier2_eval.py` (loudness normalization on, same pretrained WavLM + prematched
  HiFi-GAN), but targets VPC's required folder structure and pseudo-speaker matching sets instead
  of Tier 2's fixed real-target-speaker conversions.

## Step-by-step plan

1. **Install the toolkit**, read `run_evaluation.py` and `configs/` directly to resolve the
   `train-clean-360` question above, and confirm the exact LibriSpeech subset/speaker/utterance
   counts VPC expects for `libri_dev_enrolls`/`libri_dev_trials_m`/`libri_dev_trials_f` and the
   `libri_test_*` equivalents (VPC defines its own dev/test enrollment-vs-trial partition of
   LibriSpeech dev-clean/test-clean, not the same partition Tier 2 built).
2. **Design the pseudo-speaker assignment scheme.** Candidate approach: for each source speaker,
   build a matching set pooled from N donor speakers (e.g. N=8-20) drawn from a background pool
   disjoint from the trial speakers, so the k-NN converter never pulls two consecutive output
   frames from the same donor consistently enough to reconstruct a single identity. Needs a
   concrete, fixed selection rule (e.g. deterministic per-source-speaker hash into the donor pool)
   so the scheme itself is reproducible and documented — not ad hoc per run.
3. **Anonymize the 6 mandatory folders** (dev/test x enrolls/trials-m/trials-f) with
   `scripts/tier3_anonymize.py`, reusing Tier 2's proven encode → kNN-match → vocode pipeline
   (loudness normalization on, per the Tier 2 loudness-bug lesson) against the new pooled
   pseudo-speaker matching sets instead of single real targets. If `train-clean-360` turns out
   mandatory (step 1), anonymize that too — flag as the dominant cost item, and budget/schedule it
   as its own separately-timed job given the scale difference from the 6 short-utterance folders.
4. **Run VPC's own `run_evaluation.py`** unmodified against the anonymized folders — this is where
   the ASV attacker-retraining (ignorant / lazy-informed / semi-informed) and WER numbers come
   from; no custom eval code needed here, unlike Tiers 1-2 where we built the eval ourselves.
5. **Report**: privacy (semi-informed attacker EER — the metric that ranks submissions) and WER
   utility, and how kNN-VC-as-anonymizer compares to VPC2024's own published baseline numbers
   (B2-B6) if those are available in `results/` in the toolkit repo.

## Verification

- Toolkit installs cleanly and its own baseline (e.g. B2/McAdams, CPU-only, fastest) runs
  end-to-end first, as a smoke test that the evaluation harness itself works before pointing it at
  kNN-VC output — isolates "our anonymization is wrong" from "we set up VPC2024 wrong."
- Pseudo-speaker assignment scheme is deterministic and documented (same source speaker always
  maps to the same donor pool across a run) so results are reproducible.
- Compare against VPC2024's own published baseline privacy/utility numbers as a sanity check that
  kNN-VC's semi-informed-attacker EER and WER land in a plausible range, not just internally
  self-consistent.

## Open questions to resolve while implementing

- Is `train-clean-360` anonymization actually mandatory for the semi-informed (ranking) attacker
  metric, or can a lazy-informed/ignorant-only reduced eval skip it? Resolve by reading the actual
  toolkit source, not the README summary used to draft this plan.
- Exact compute budget once (1) is resolved — this is the single biggest unknown in scoping this
  tier's time/cost, potentially far larger than Tiers 1-2 combined.
- What N (donor pool size) and selection rule for pseudo-speaker assignment gives a defensible
  privacy/utility tradeoff — this is genuinely open design work, worth a small ablation (vary N,
  check EER vs. WER) rather than picking one value blind.
- Whether VPC2024 has any published numbers for a kNN-VC-like baseline already (STTTS, B3, uses a
  not-dissimilar phone-sequence + speaker-embedding approach) that could sanity-check expected
  EER/WER ranges before running the full pipeline.
