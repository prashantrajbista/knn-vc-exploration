"""Tier 2: objective WER/CER/EER reproduction of Table 1 (kNN-VC row + topline),
using only pretrained checkpoints (WavLM, HiFi-GAN prematched, Whisper-base, x-vector).
No training. See docs/plan_tier_2.md for the full protocol writeup.

Usage:
    python scripts/tier2_eval.py --sanity          # 2 speakers, 2 conversions, fast pipeline check
    python scripts/tier2_eval.py                   # full paper protocol: 40 speakers x 5 utts x 39 targets
"""
import argparse
import json
import logging
import random
import time
from pathlib import Path

import jiwer
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import roc_curve
from torchaudio.datasets import LIBRISPEECH

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "notebooks" / "data"  # reuse Tier 1's LibriSpeech cache location
OUT_ROOT = REPO_ROOT / "data" / "tier2_outputs"

# Table 1 (paper, LibriSpeech test-clean) for the final comparison printout.
PAPER_KNNVC = {"wer": 7.36, "cer": 2.96, "eer": 37.15}
PAPER_TOPLINE = {"wer": 5.96, "cer": 2.38}

log = logging.getLogger("tier2")

# LibriSpeech transcripts are upper-case with no punctuation ("THE QUICK BROWN FOX");
# Whisper output is mixed-case with punctuation ("The quick brown fox."). Without
# normalizing both to the same convention, jiwer counts every word as a substitution.
_WORD_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(), jiwer.ReduceToListOfListOfWords(),
])
_CHAR_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.Strip(), jiwer.ReduceToListOfListOfChars(),
])


def wer(refs, hyps):
    return jiwer.wer(refs, hyps, reference_transform=_WORD_TRANSFORM, hypothesis_transform=_WORD_TRANSFORM)


def cer(refs, hyps):
    return jiwer.cer(refs, hyps, reference_transform=_CHAR_TRANSFORM, hypothesis_transform=_CHAR_TRANSFORM)


def setup_logging(out_dir, verbose=False):
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    log.addHandler(stream)
    file_handler = logging.FileHandler(out_dir / "tier2.log")
    file_handler.setFormatter(fmt)
    log.addHandler(file_handler)


def eta_str(n_new, n_remaining, elapsed):
    """n_new = items actually computed so far (skips don't count, they're ~instant)."""
    rate = n_new / elapsed if elapsed > 0 else 0
    if rate == 0:
        return "rate unknown"
    return f"{rate:.2f}/s, ~{n_remaining / rate / 60:.1f} min left"


def get_device():
    # knn-vc's hub loader only handles cuda->cpu (hifigan_wavlm doesn't check mps),
    # so mixing mps here would split encoder/vocoder across devices. Same fallback
    # used in notebooks/01_tier1_inference.ipynb.
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_eval_set(dataset, n_speakers, n_per_speaker):
    """Deterministic 40-speakers x 5-utterances selection (paper doesn't publish its
    exact sample/seed, so this is *a* reproducible sample, not necessarily identical)."""
    speaker_to_indices = {}
    for i in range(len(dataset)):
        _, _, _, speaker_id, _, _ = dataset.get_metadata(i)
        speaker_to_indices.setdefault(speaker_id, []).append(i)
    for idxs in speaker_to_indices.values():
        idxs.sort()

    speakers = sorted(speaker_to_indices)[:n_speakers]
    eval_indices = {spk: speaker_to_indices[spk][:n_per_speaker] for spk in speakers}
    return speaker_to_indices, speakers, eval_indices


def matching_indices_for_speaker(speaker_to_indices, eval_indices, speaker):
    """All of a speaker's utterances except the ones reserved as eval sources for
    that speaker -- avoids leaking an eval utterance into its own target matching set."""
    excluded = set(eval_indices.get(speaker, []))
    return [i for i in speaker_to_indices[speaker] if i not in excluded]


def convert_all(knn_vc, dataset, speakers, eval_indices, speaker_to_indices, n_targets, k, out_dir, device):
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    matching_set_cache = {}  # target speaker -> (matching_set tensor, matching wav indices used)
    query_cache = {}  # (speaker, idx) -> features tensor

    def get_matching_set(speaker):
        if speaker not in matching_set_cache:
            idxs = matching_indices_for_speaker(speaker_to_indices, eval_indices, speaker)
            wavs = [dataset[i][0] for i in idxs]
            matching_set_cache[speaker] = (knn_vc.get_matching_set(wavs), idxs)
        return matching_set_cache[speaker]

    def get_query(speaker, idx):
        key = (speaker, idx)
        if key not in query_cache:
            wave, *_ = dataset[idx]
            query_cache[key] = knn_vc.get_features(wave)
        return query_cache[key]

    n_total = sum(min(n_targets, len(speakers) - 1) * len(eval_indices[s]) for s in speakers)
    n_done, n_new, n_skipped = 0, 0, 0
    start = time.time()
    log.info(f"conversion: {n_total} (source, target) pairs to produce, output dir {wav_dir}")

    for src_speaker in speakers:
        targets = [s for s in speakers if s != src_speaker][:n_targets]
        for src_idx in eval_indices[src_speaker]:
            query_seq = None
            for tgt_speaker in targets:
                out_path = wav_dir / f"{src_speaker}-{src_idx}__to__{tgt_speaker}.wav"
                if out_path.exists():
                    n_done += 1
                    n_skipped += 1
                    continue
                if query_seq is None:
                    log.debug(f"encoding source {src_speaker}-{src_idx}")
                    query_seq = get_query(src_speaker, src_idx)
                if tgt_speaker not in matching_set_cache:
                    log.info(f"building matching set for target speaker {tgt_speaker} (first use)")
                matching_set, _ = get_matching_set(tgt_speaker)
                # tgt_loudness_db left at knn_vc.match's default (-16 dB LUFS) -- Tier 1's
                # notebook disabled it only to diff against a hand-rolled pipeline that
                # also skipped normalization; nothing here calls for skipping it.
                wav = knn_vc.match(query_seq, matching_set, topk=k).cpu()
                torchaudio.save(str(out_path), wav[None], knn_vc.sr)
                n_done += 1
                n_new += 1
                if n_new % 25 == 0:
                    elapsed = time.time() - start
                    log.info(f"converted {n_done}/{n_total} ({n_skipped} skipped, already existed) -- {eta_str(n_new, n_total - n_done, elapsed)}")

    log.info(f"conversion stage done: {n_done}/{n_total} outputs ({n_new} newly computed, {n_skipped} already existed) in {wav_dir}")
    return matching_set_cache


def run_asr(whisper_model, dataset, speakers, eval_indices, out_dir):
    wav_dir = out_dir / "wavs"
    transcripts = {}
    for speaker in speakers:
        for idx in eval_indices[speaker]:
            _, _, text, *_ = dataset[idx]
            transcripts[(speaker, idx)] = text

    wav_paths = sorted(wav_dir.glob("*__to__*.wav"))
    log.info(f"ASR: transcribing {len(wav_paths)} converted utterances")
    start = time.time()
    refs, hyps = [], []
    for i, wav_path in enumerate(wav_paths, 1):
        src_part, _ = wav_path.stem.split("__to__")
        src_speaker, src_idx = src_part.rsplit("-", 1)
        key = (int(src_speaker), int(src_idx))
        ref = transcripts[key]
        hyp = whisper_model.transcribe(str(wav_path))["text"]
        refs.append(ref)
        hyps.append(hyp)
        if i <= 5:
            log.info(f"[converted {i}] {wav_path.name}\n    ref: {ref}\n    hyp: {hyp}")
        if i % 25 == 0:
            log.info(f"transcribed {i}/{len(wav_paths)} converted -- {eta_str(i, len(wav_paths) - i, time.time() - start)}")

    converted_wer = 100 * wer(refs, hyps)
    converted_cer = 100 * cer(refs, hyps)
    log.info(f"converted: WER {converted_wer:.2f} CER {converted_cer:.2f}")

    # Topline: Whisper directly on the original, unconverted source utterances.
    n_topline = sum(len(eval_indices[s]) for s in speakers)
    log.info(f"ASR: transcribing {n_topline} original (unconverted) source utterances for topline")
    start = time.time()
    top_refs, top_hyps = [], []
    tmp_path = out_dir / "_topline_tmp.wav"
    i = 0
    for speaker in speakers:
        for idx in eval_indices[speaker]:
            wave, sr, text, *_ = dataset[idx]
            torchaudio.save(str(tmp_path), wave, sr)
            top_refs.append(text)
            top_hyp = whisper_model.transcribe(str(tmp_path))["text"]
            top_hyps.append(top_hyp)
            i += 1
            if i <= 5:
                log.info(f"[topline {i}] speaker {speaker} idx {idx}\n    ref: {text}\n    hyp: {top_hyp}")
            if i % 25 == 0:
                log.info(f"transcribed {i}/{n_topline} topline -- {eta_str(i, n_topline - i, time.time() - start)}")
    if tmp_path.exists():
        tmp_path.unlink()

    topline_wer = 100 * wer(top_refs, top_hyps)
    topline_cer = 100 * cer(top_refs, top_hyps)
    log.info(f"topline: WER {topline_wer:.2f} CER {topline_cer:.2f}")

    return {"wer": converted_wer, "cer": converted_cer}, {"wer": topline_wer, "cer": topline_cer}


def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = (fpr - fnr).__abs__().argmin()
    return 100 * (fpr[idx] + fnr[idx]) / 2


def run_eer(spk_model, dataset, speakers, eval_indices, speaker_to_indices, out_dir, device, seed=42):
    wav_dir = out_dir / "wavs"
    rng = random.Random(seed)

    def embed(wave):
        with torch.no_grad():
            emb = spk_model(wave.to(device))  # (1, samples) -> (1, 256), already unit-length
        return emb.squeeze(0).cpu()

    log.info(f"EER: scoring against {len(speakers)} target speakers")
    start = time.time()
    scores, labels = [], []
    for i, tgt_speaker in enumerate(speakers, 1):
        converted = sorted(wav_dir.glob(f"*__to__{tgt_speaker}.wav"))
        if not converted:
            log.debug(f"speaker {tgt_speaker}: no converted samples found, skipping")
            continue

        matching_idxs = matching_indices_for_speaker(speaker_to_indices, eval_indices, tgt_speaker)
        if len(matching_idxs) < 2:
            log.warning(f"speaker {tgt_speaker}: not enough non-eval utterances for enrollment, skipping")
            continue
        enroll_idx = matching_idxs[0]
        genuine_pool = matching_idxs[1:]

        enroll_wave, *_ = dataset[enroll_idx]
        enroll_emb = embed(enroll_wave)

        for wav_path in converted:
            wav, sr = torchaudio.load(str(wav_path))
            conv_emb = embed(wav)
            scores.append(F.cosine_similarity(enroll_emb, conv_emb, dim=0).item())
            labels.append(0)  # converted / impostor-side of the paper's protocol

            genuine_idx = rng.choice(genuine_pool)
            genuine_wave, *_ = dataset[genuine_idx]
            genuine_emb = embed(genuine_wave)
            scores.append(F.cosine_similarity(enroll_emb, genuine_emb, dim=0).item())
            labels.append(1)  # genuine target-speaker pair

        log.info(f"speaker {i}/{len(speakers)} ({tgt_speaker}) done, {len(scores)} scores so far -- {eta_str(i, len(speakers) - i, time.time() - start)}")

    eer = compute_eer(labels, scores)
    log.info(f"EER: {eer:.2f}% over {len(scores)} scores")
    return eer, len(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test-clean")
    parser.add_argument("--n-speakers", type=int, default=40)
    parser.add_argument("--n-per-speaker", type=int, default=5)
    parser.add_argument("--n-targets", type=int, default=None, help="default: all other eval speakers (39 for full run)")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--stage", choices=["convert", "asr", "eer", "all"], default="all")
    parser.add_argument("--out-dir", default=str(OUT_ROOT))
    parser.add_argument("--sanity", action="store_true", help="tiny 2-speaker/1-utt/1-target smoke test")
    parser.add_argument("--verbose", action="store_true", help="DEBUG-level logging (per-utterance detail)")
    args = parser.parse_args()

    if args.sanity:
        args.n_speakers, args.n_per_speaker, args.n_targets = 2, 1, 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir, verbose=args.verbose)
    log.info(f"args: {vars(args)}")

    device = get_device()
    log.info(f"device: {device}")

    log.info(f"loading LibriSpeech {args.split} (auto-downloads to {DATA_ROOT} if missing)...")
    dataset = LIBRISPEECH(root=str(DATA_ROOT), url=args.split, download=True)
    speaker_to_indices, speakers, eval_indices = build_eval_set(dataset, args.n_speakers, args.n_per_speaker)
    n_targets = args.n_targets if args.n_targets is not None else len(speakers) - 1
    log.info(f"eval set: {len(speakers)} speakers x {args.n_per_speaker} utterances, {n_targets} targets each")

    results = {}

    if args.stage in ("convert", "all"):
        log.info("loading pretrained kNN-VC pipeline (WavLM + prematched HiFi-GAN)...")
        knn_vc = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=device)
        log.info("starting conversion stage")
        convert_all(knn_vc, dataset, speakers, eval_indices, speaker_to_indices, n_targets, args.k, out_dir, device)

    if args.stage in ("asr", "all"):
        log.info(f"loading Whisper-{args.whisper_model}...")
        import whisper
        whisper_model = whisper.load_model(args.whisper_model, device=device)
        log.info("starting ASR stage")
        converted, topline = run_asr(whisper_model, dataset, speakers, eval_indices, out_dir)
        results["converted_wer_cer"] = converted
        results["topline_wer_cer"] = topline

    if args.stage in ("eer", "all"):
        # RF5/simple-speaker-embedding: GE2E speaker embedding model released by kNN-VC's
        # first author (Matthew Baas / "rf5"), trained on VCTK+LibriSpeech+VoxCeleb1+2,
        # self-reported LibriSpeech test-clean EER 2.95% -- matches the exact domain/metric
        # Table 1 needs, and is a far closer proxy to the paper's own eval setup than a
        # generic VoxCeleb x-vector.
        log.info("loading pretrained speaker embedding model (RF5/simple-speaker-embedding)...")
        spk_model = torch.hub.load("RF5/simple-speaker-embedding", "convgru_embedder", trust_repo=True, device=device)
        spk_model.eval()
        log.info("starting EER stage")
        eer, n_scores = run_eer(spk_model, dataset, speakers, eval_indices, speaker_to_indices, out_dir, device)
        results["eer"] = eer
        results["n_eer_scores"] = n_scores

    results_path = out_dir / "results.json"
    if results_path.exists():
        prior = json.loads(results_path.read_text())
        prior.update(results)
        results = prior
    results_path.write_text(json.dumps(results, indent=2))

    print("\n--- Tier 2 results vs. paper Table 1 (LibriSpeech test-clean) ---")
    if "converted_wer_cer" in results:
        c = results["converted_wer_cer"]
        print(f"kNN-VC   WER {c['wer']:.2f} (paper {PAPER_KNNVC['wer']})   CER {c['cer']:.2f} (paper {PAPER_KNNVC['cer']})")
    if "topline_wer_cer" in results:
        t = results["topline_wer_cer"]
        print(f"topline  WER {t['wer']:.2f} (paper {PAPER_TOPLINE['wer']})   CER {t['cer']:.2f} (paper {PAPER_TOPLINE['cer']})")
    if "eer" in results:
        print(f"kNN-VC   EER {results['eer']:.2f}% (paper {PAPER_KNNVC['eer']}%)  [n={results['n_eer_scores']} scores]")
    print(f"\nfull results: {results_path}")


if __name__ == "__main__":
    main()
