# Voice Conversion With Just Nearest Neighbors

**Metadata**

- Authors / year / venue: Matthew Baas, Benjamin van Niekerk, Herman Kamper (MediaLab, Stellenbosch University). Interspeech 2023.
- Link / DOI / arXiv: arXiv:2305.18975v1 [eess.AS], 30 May 2023
- Code available? (URL, framework, last commit): Yes — code, samples, trained models at https://bshall.github.io/knn-vc (PyTorch). Pretrained WavLM-Large encoder + HiFi-GAN vocoder checkpoints released.
- Dataset(s): LibriSpeech dev-clean and test-clean (40 speakers each, ~8 min/speaker, 16 kHz English) for evaluation; LibriSpeech train-clean-100 for HiFi-GAN vocoder training.
- Date I read it / status: 2026-08-10 / pass 2
- Tags: #voice-conversion #kNN #self-supervised #WavLM #HiFi-GAN

---

## PASS 1 — The 5-to-10-minute skim

_Read: title, abstract, intro, section headings, figures, conclusion. Goal: decide if it's worth more of your time and place it on the map._

- **Category** — New method paper (any-to-any voice conversion), deliberately minimal/non-parametric — positioned as a simplicity baseline against complex neural VC systems.
- **Context** — Responds to the any-to-any VC line (VQMIVC, FreeVC, YourTTS, AutoVC) that uses learned disentanglement (VAE, VQ+mutual-information, normalization, text bottlenecks) to separate speaker from content. Builds on self-supervised speech representation work (wav2vec 2.0, WavLM) and layer-wise probing studies showing SSL features linearly encode phonetic identity. Also revives classic concatenative VC (unit selection), replacing hand-defined units with SSL feature frames.
- **Correctness** — Assumptions are testable and the paper stress-tests its own key design choice (which WavLM layer to use) empirically rather than assuming it; sound.
- **Contributions** — In their own words: (1) kNN-VC — simple, training-free (for the conversion step) any-to-any VC method using kNN regression over self-supervised features; (2) prematched vocoder training to close the train/inference mismatch; (3) empirical demonstration that kNN-VC matches or beats complex baselines on intelligibility and speaker similarity; (4) ablations on target data size and prematching; (5) release of a reproducible, easy-to-implement baseline for the field.
- **Clarity** — Very clear, short (5 pages), one clean figure (encoder-converter-vocoder diagram) carries the whole method. Easy to reproduce from the paper alone.
- **One-sentence summary** in _my own_ words: Instead of training a model to disentangle speaker from content, kNN-VC extracts WavLM layer-6 features for source and target speech and replaces each source frame with the average of its k=4 nearest target frames (cosine distance), then vocodes the result with a HiFi-GAN trained to expect exactly this kind of nearest-neighbor-averaged input — and this "dumb" non-parametric swap beats or matches state-of-the-art trained VC systems on speaker similarity while staying just as intelligible.
- **Verdict** — must reproduce (dedicated repo for this paper; method is training-free at the conversion step, so reproducing the pipeline only requires the frozen WavLM encoder and a HiFi-GAN vocoder, either pretrained or fine-tuned).

---

## PASS 2 — Grasp the content, not every detail (Keshav pass 2 + QALMRI)

**Q — Question**

- Is architectural/training complexity actually necessary for high-quality any-to-any voice conversion, or can a simple non-parametric method built on modern self-supervised speech representations match or beat trained disentanglement models? Prior any-to-any systems improve naturalness/similarity but at increasing cost in bottleneck design, normalization tricks, data augmentation, or text supervision — making them hard to reproduce and build on.

**A — Alternatives**

- VQMIVC — vector quantization + mutual-information minimization to disentangle speaker from content (trained model).
- FreeVC — VAE + data augmentation to strip speaker information (trained model); the strongest baseline on naturalness/intelligibility.
- YourTTS — uses text transcriptions as an intermediate bottleneck for disentanglement (zero-shot multi-speaker TTS repurposed for VC); needs transcribed training data.
- AutoVC — autoencoder-only zero-shot style transfer (discussed but not directly benchmarked in Table 1).
- All three benchmarked baselines condition on a speaker-embedding model (averaged over reference utterances); kNN-VC has no such embedding model at all.

**L — Logic**

- Self-supervised speech models (WavLM, wav2vec 2.0) already encode phonetic content in a way that's locally linear/metric — frames of the same phone cluster close together in feature space regardless of speaker. If that's true, you don't need to *learn* a disentangled space: for every source frame, just look up the nearest target-speaker frame with the same phonetic content and substitute it directly. Content is preserved because the substitution is nearest-neighbor (same phone), and speaker identity is preserved because the substituted vector is literally drawn from the target speaker's own recordings (concatenative-style guarantee).

**M — Method**

- Architecture / algorithm: encoder → converter → vocoder pipeline. (1) Encoder: pretrained WavLM-Large, frozen, feature extracted from **layer 6** specifically (not later layers 22/24, which are better for phone recognition but worse for pitch/energy/speaker-identity retention). Produces one 1024-dim vector per 20ms. (2) Converter: for every query (source) frame, find its k=4 nearest neighbors (cosine distance) in the "matching set" (pooled, shuffled WavLM frames from all available target-speaker reference utterances) and replace the query frame with their unweighted mean. No training, no learned parameters in this step. (3) Vocoder: HiFi-GAN V1, modified to take the 1024-dim WavLM vectors as input instead of mel-spectrograms (still targets 128-dim mel internally at 10ms hop / 64ms Hann window, 16kHz output), trained on LibriSpeech train-clean-100.
- Key design choices and _why_: **Layer 6 of WavLM-Large** — chosen after ablating later layers empirically; earlier/mid layers trade off phone-discriminability against pitch/prosody/speaker-identity retention, and layer 6 was the sweet spot found here. **k=4, uniform weighting, cosine distance** — robust to a range around 4; can go up to k≈20 with ≥10 min of reference audio for slightly better quality. **Prematched vocoder training** (Section 3.4) — reconstruct the HiFi-GAN training set itself via kNN (map each training utterance's frames to its k nearest neighbors from *other* utterances of the *same* speaker) so the vocoder trains on data that already looks like noisy/averaged kNN output rather than clean WavLM features — closes a train/inference mismatch and gives a "non-negligible" EER/WER improvement at every reference-data size (Fig. 2). No speaker embedding model anywhere in the pipeline — a structural difference from every baseline.
- Assumptions made (explicit and hidden): Explicit — SSL representations encode phonetic similarity as feature-space proximity (load-bearing; if it doesn't hold in a different feature space, layer, or language, the whole method degrades). Hidden/untested: WavLM was pretrained on largely English/Western speech, so the layer-6 choice and the phonetic-proximity assumption are only validated in-distribution (LibriSpeech English). The kNN-matching-set approach also implicitly assumes the matching set covers the phone/biphone inventory needed for the source utterance — degrades sharply below ~30s of reference audio (Fig. 2).
- Training setup: Only the vocoder is trained (encoder is frozen pretrained WavLM, converter is training-free kNN). HiFi-GAN V1 architecture, same optimizer/steps/hyperparameters as the original HiFi-GAN paper, trained on LibriSpeech train-clean-100 (16kHz, 128-dim mel target, 10ms hop, 64ms Hann window), two variants (plain WavLM features vs. prematched features). Inference with 8 min of reference audio runs faster than real-time on a consumer 8GB VRAM GPU.

**R — Results**

- Headline result(s) and the metric(s) used: On LibriSpeech test-clean (Table 1) — kNN-VC: WER 7.36, CER 2.96, EER 37.15%, MOS 4.03±0.08, SIM 2.91±0.11. Best baseline FreeVC: WER 7.61, CER 3.17, EER 8.97%, MOS 4.07±0.07, SIM 2.38±0.11. Testset topline (real, unconverted speech): WER 5.96, CER 2.38, MOS 4.24±0.07, SIM 3.19±0.09. kNN-VC essentially matches FreeVC on intelligibility/naturalness but **more than quadruples** its speaker-similarity EER (37.15% vs 8.97%; max possible is 50% = indistinguishable from genuine target speech) and clearly beats it on subjective SIM too.
- Key figures/tables: Table 1 is the central results table (comparison to 3 SOTA baselines + topline). Figure 2 (WER vs. EER scatter across target-data sizes: 5s/10s/30s/1m/5m/8m, for plain vs. prematched HiFi-GAN) carries the ablation claim — prematching shifts the whole curve toward the upper-left (better) at every data size, and performance gains saturate around 5 minutes of reference audio.
- Ablations they ran: (1) SSL layer choice (6 vs. 22/24 vs. mean-of-last-several) — informal, reported in prose not a table. (2) k value sensitivity — informal, found robust around k=4, better with k≈20 given ≥10 min reference. (3) Prematched vs. plain vocoder training × target-data size (5s to 8m) — the main formal ablation, Figure 2. (4) Comparison of "plain HiFi-GAN, no prematching" numbers against Table 1 baselines, showing even the weaker kNN-VC variant is still competitive/superior — isolates how much of the gain is from kNN-matching itself vs. from prematching.

**I — Inferences**

- What do the authors conclude? Complexity is not necessary for high-quality any-to-any voice conversion — a training-free kNN swap over frozen self-supervised features, plus a vocoder adapted to expect that specific kind of input (prematching), rivals or beats purpose-trained disentanglement models, particularly on speaker similarity. They position kNN-VC as a new easy-to-reproduce baseline for the field, and note (without deep investigation) that it may generalize to cross-lingual and even non-speech conversion since it makes no speaker-embedding-model assumption.
- Do the results support the claim, or is there a gap? Support is strong on LibriSpeech (English, read speech, clean studio-ish audio, 40 held-out speakers, ~8 min/speaker) — a fairly favorable setting for WavLM (itself likely pretrained on largely English/Western data). The paper is explicit that cross-lingual/cross-domain robustness is only shown qualitatively on the demo page (German→Japanese conversion, whispered speech, non-speech), not quantitatively evaluated — a real gap between the claim of generality and the evidence given.

**Unknowns to look up** _(PLOS Rule: don't skip what you don't understand)_

- WavLM pretraining corpus composition (language distribution) — relevant to how much the layer-6 phonetic-similarity property is English-specific.
- Whether other SSL encoders (e.g. XLS-R, HuBERT) exhibit the same "one mid-layer is best for speaker/prosody" property, or whether the specific layer index is WavLM-specific and needs re-discovering per encoder.
- Prematched training implementation details beyond the paper (exact same-speaker matching-set construction at training time, edge cases with very few training utterances per speaker) — check the public repo.
- Van Niekerk et al. 2022 (ICASSP), "A comparison of discrete and soft speech units for improved voice conversion" — cited as the EER speaker-similarity evaluation protocol source (ref [22]), same authors, likely useful for implementing the EER-via-x-vector-cosine-similarity metric.

---

## Synthesis _(the part that compounds — Zettelkasten)_

- **In 3 sentences, for a peer who hasn't read it:** kNN-VC converts any source speaker to any target speaker by extracting WavLM layer-6 features from both, replacing each source frame with the mean of its 4 nearest target frames (cosine distance), and vocoding the result with a HiFi-GAN specially trained on this same kind of "prematched" nearest-neighbor-averaged input. Despite having zero trained parameters in the actual conversion step, it beats three SOTA trained any-to-any VC systems (VQMIVC, FreeVC, YourTTS) on speaker similarity (EER 37% vs. 9-25%) while matching them on intelligibility and naturalness, on LibriSpeech test-clean. The authors argue this shows disentanglement doesn't need to be *learned* — modern self-supervised features already linearize phonetic content well enough that nearest-neighbor lookup is sufficient.
- **Strongest idea I want to keep:** The prematched vocoder training trick — training the vocoder on the *exact statistical distribution of its own inference-time input* (kNN-averaged features, not clean features) rather than assuming train/inference distributions match — is a small, cheap, generally reusable idea beyond VC (applies to any generation model that will only ever see post-processed/averaged conditioning at inference).
- **Weakest link / what I'd challenge:** Everything is validated on English/LibriSpeech only; the cross-lingual claims are relegated to unquantified demo-page examples, not evaluated with WER/EER/MOS. The method's core assumption — that a fixed mid-layer of an English-heavy SSL model linearizes phonetic content well for any language/domain — is untested at the quantitative level. Also: no speaker-embedding model means kNN-VC's behavior as an anonymization technique (rather than just conversion quality) is unexamined here — the paper only measures conversion quality/similarity, not adversarial re-identification risk.
- **My original angle** — a question the paper doesn't answer that I could turn into a study: Does the WavLM-layer-6 "speaker-identity-correlated layer" choice hold on out-of-English or noisy/telephony speech, or does the optimal layer shift when the encoder's pretraining data doesn't cover the target domain well? A small layer-sweep ablation (repeat the informal layer-6-vs-22-vs-24 comparison from this paper on a different language/domain, using EER/WER as the readout) would directly test the core transferability assumption before defaulting to layer 6.
- **Links to other papers I've read** (builds on ___ / contradicts ___ / extends ___):
- **If I cite this later, what for:** As the core architecture/method reference (WavLM layer 6 + kNN regression k=4 + prematched HiFi-GAN) when reproducing or extending kNN-VC, and specifically for the prematched-training technique if reused elsewhere. Also as evidence that non-parametric, training-light methods are a legitimate SOTA-competitive choice for voice conversion when compute/data for a full trained disentanglement model isn't available.
