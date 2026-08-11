# Resource estimate for reproducing kNN-VC

Three tiers, since "reproduce" can mean different depths.

## Tier 1 — inference-only

Use released pretrained WavLM + HiFi-GAN checkpoints, just run conversion.

- GPU: 8GB VRAM consumer card (paper's own claim), faster than real-time.
- Storage: ~2GB (WavLM-Large ~1.26GB + HiFi-GAN checkpoint ~50-100MB + a few test utterances).
- Time: hours — download checkpoints, run demo conversions, sanity-check against their samples page.
- Cost: ~$0, or <$5 on rented GPU if no local GPU.

## Tier 2 — train vocoder yourself + reproduce Fig. 2 ablation

Plain vs prematched HiFi-GAN, target-data-size sweep.

- Data: LibriSpeech train-clean-100 (~6.3GB, 100hrs) for vocoder training; dev-clean (~340MB) for the ablation sweep.
- Compute: HiFi-GAN V1 is known to need ~2.5M steps to converge — roughly 3-5 days on a single modern GPU (3090/4090/A100-class) per variant. Two variants (plain + prematched) → ~1-2 weeks sequential, ~3-5 days if parallel on 2 GPUs.
- Extra preprocessing: WavLM feature extraction over 100hrs (GPU inference, ~1hr) + prematching kNN construction (CPU/GPU matrix ops, fast, minutes-hours depending on implementation).
- Storage: +6.3GB data, +checkpoints during training (a few GB, HiFi-GAN saves intermediate ckpts).
- Cost (cloud GPU rental ~$1/hr): roughly $150-250 for both variants.

## Tier 3 — full paper reproduction incl. Table 1 baselines + subjective eval

- Baselines: VQMIVC, FreeVC, YourTTS — all inference-only (pretrained public checkpoints), no training needed. Few hundred MB each, minor compute.
- Objective metrics: Whisper-base ASR (WER/CER) + x-vector speaker verification (EER) — both pretrained, inference only, cheap.
- Subjective MOS/SIM: needs human raters (paper used Mechanical Turk, ~3800 total ratings). This is the expensive/slow part — budget ~$200-500 and days of turnaround if replicating this piece; skip if only the technical claim matters (speaker similarity via EER already gives an objective proxy).

## Bottom line

If goal is "verify the method works," Tier 1 is basically free and same-day. If goal is "reproduce the paper's quantitative claims," budget Tier 2's ~1-2 weeks + ~$200 GPU cost, skip subjective eval unless MOS/SIM numbers specifically need matching.
