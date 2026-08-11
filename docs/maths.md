# Math concepts for reproducing kNN-VC

Grouped by pipeline stage. Priority for reproduction noted at bottom.

## 1. Vector space + distance metrics (core of kNN step)

- Feature vectors: each 20ms frame → vector in ℝ¹⁰²⁴ (WavLM layer 6 output).
- Cosine distance: `d(x,y) = 1 - (x·y)/(‖x‖‖y‖)`. Paper uses this, not Euclidean — measures angle, not magnitude. Angle is more robust than magnitude for "same phonetic content" comparison since speaker/loudness scale features differently.
- k-NN regression (Fix & Hodges, nonparametric): for query x, find k nearest y_i in reference set by distance, output ŷ = (1/k)Σy_i (uniform weight, unweighted mean — no kernel weighting despite "regression" name). This is the whole converter step, zero learned params.
- Nearest-neighbor search cost: brute-force cosine search over matching set is O(N) per query vector. 8 min reference audio ≈ 24k frames — matters for implementation speed.

## 2. Self-supervised representation learning (encoder, WavLM)

- Multi-head self-attention: `softmax(QKᵀ/√d)V`, standard transformer block.
- Masked prediction pretraining objective: predict masked frame's pseudo-label from context. WavLM specifically adds denoising — mixes in noise/overlapping speech during pretraining.
- Layer-wise probing: different layers encode different linearly-decodable info (phonetic vs speaker vs prosody). Math here is just linear probing — train logistic regression on frozen features, measure accuracy. Explains why layer 6 ≠ layer 22 for this task.

## 3. Signal processing (mel-spectrogram, vocoder I/O)

- STFT: `X(t,f) = Σ_n x[n] w[n-t] e^{-j2πfn/N}` — windowed short-time Fourier transform, gives time-frequency representation.
- Mel-scale: `m = 2595 log10(1 + f/700)` — nonlinear frequency warp mimicking human pitch perception, used for 128-dim mel target in HiFi-GAN.
- Hann window: `w[n] = 0.5(1-cos(2πn/(N-1)))` — smooths frame edges before FFT, reduces spectral leakage. Paper uses 64ms window / 10ms hop — must match exactly for correct vocoder I/O.

## 4. GAN objective (HiFi-GAN vocoder training)

- Adversarial loss (LSGAN-style, not vanilla BCE):
  - `L_D = E[(D(x)-1)² + D(G(z))²]`
  - `L_G = E[(D(G(z))-1)²]`
- Feature matching loss: L1 distance between discriminator intermediate activations of real vs generated audio, `L_FM = Σ_i (1/N_i)‖D_i(x) - D_i(G(z))‖₁`.
- Mel-spectrogram reconstruction loss: L1 between mel(x) and mel(G(z)) — direct spectral supervision, dominant loss term in practice.
- Multi-period + multi-scale discriminator: several discriminators at different sample-rate strides/scales, sum their losses. This is HiFi-GAN's actual architectural novelty.

## 5. Evaluation metrics (needed to verify reproduction matches paper numbers)

- WER/CER: edit (Levenshtein) distance normalized by reference length — `WER = (S+D+I)/N` (substitutions+deletions+insertions / reference word count).
- EER (equal error rate): point on ROC curve where false-accept rate = false-reject rate, computed from cosine-similarity scores between x-vector speaker embeddings. Requires ROC/threshold-sweep — plot FAR(θ) and FRR(θ) over threshold θ, find crossing.
- x-vector: pretrained DNN speaker-embedding model, trained via classification (softmax over speaker IDs) then pooled — reused as a black box, not trained in this paper.

## Priority for reproduction

- **Strictly need**: #1 (kNN + cosine, trivial) and #3 (mel-spectrogram config — must match exactly for vocoder input/output).
- **Treat as pretrained black box**: #2 and #4, unless fine-tuning the vocoder yourself — then #4 (GAN losses) becomes mandatory.
- **Only needed for exact benchmark reproduction**: #5, not needed for just running the demo pipeline.
