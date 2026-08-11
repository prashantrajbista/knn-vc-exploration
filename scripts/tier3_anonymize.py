"""Tier 3 step 3: anonymize VPC2024's 6 mandatory LibriSpeech folders with kNN-VC, using
pooled multi-donor pseudo-speaker matching sets (tier3_pseudo_speaker.py) instead of the
paper's fixed single real target speaker.

Prerequisite: scripts/tier3_prepare_data.py has populated
<vpc-root>/data/libri_{dev,test}/wav/<uttid>/<uttid>.wav.

Output layout matches what VPC2024's run_evaluation.py expects for a custom anonymization
system (README "Custom Anonymization Integration"):
    <vpc-root>/data/libri_dev_enrolls${suffix}/wav/*.wav
    <vpc-root>/data/libri_dev_trials_m${suffix}/wav/*.wav
    <vpc-root>/data/libri_dev_trials_f${suffix}/wav/*.wav
    <vpc-root>/data/libri_test_enrolls${suffix}/wav/*.wav
    <vpc-root>/data/libri_test_trials_m${suffix}/wav/*.wav
    <vpc-root>/data/libri_test_trials_f${suffix}/wav/*.wav

Usage:
    python scripts/tier3_anonymize.py --sanity   # 5-utterance smoke test
    python scripts/tier3_anonymize.py            # full run: 4255 utterances
"""
import argparse
import logging
import time
from pathlib import Path

import torch
import torchaudio

from tier3_pseudo_speaker import assign_donors, load_speaker_pool

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VPC_ROOT = REPO_ROOT / "third_party" / "Voice-Privacy-Challenge-2024"

FOLDERS = [
    ("dev", "enrolls"), ("dev", "trials_m"), ("dev", "trials_f"),
    ("test", "enrolls"), ("test", "trials_m"), ("test", "trials_f"),
]

log = logging.getLogger("tier3")


def setup_logging(out_log_path: Path, verbose=False):
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    log.addHandler(stream)
    file_handler = logging.FileHandler(out_log_path)
    file_handler.setFormatter(fmt)
    log.addHandler(file_handler)


def eta_str(n_done, n_remaining, elapsed):
    rate = n_done / elapsed if elapsed > 0 else 0
    if rate == 0:
        return "rate unknown"
    return f"{rate:.2f}/s, ~{n_remaining / rate / 60:.1f} min left"


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_utt_path_index(vpc_data_dir: Path) -> dict[str, Path]:
    """utt_id -> wav path, across both reconstructed raw pools."""
    index = {}
    for split in ("dev", "test"):
        wav_root = vpc_data_dir / f"libri_{split}" / "wav"
        for utt_dir in wav_root.iterdir():
            index[utt_dir.name] = utt_dir / f"{utt_dir.name}.wav"
    return index


def folder_utterances(vpc_data_dir: Path, split: str, subset: str) -> list[tuple[str, str]]:
    """[(utt_id, true_speaker), ...] for one of the 6 mandatory folders, from VPC's own
    utt2spk metadata (authoritative over parsing the utt_id, though they agree)."""
    utt2spk_path = vpc_data_dir / f"libri_{split}_{subset}" / "utt2spk"
    pairs = []
    for line in utt2spk_path.read_text().splitlines():
        utt_id, speaker = line.split()
        pairs.append((utt_id, speaker))
    return pairs


def anonymize_all(knn_vc, vpc_data_dir: Path, utt_path_index: dict[str, Path],
                   speaker_pool: dict[str, list[str]], suffix: str, k: int, limit: int | None):
    matching_set_cache: dict[tuple, torch.Tensor] = {}
    donor_feature_cache: dict[str, torch.Tensor] = {}

    def get_donor_features(utt_id: str) -> torch.Tensor:
        if utt_id not in donor_feature_cache:
            wav, sr = torchaudio.load(str(utt_path_index[utt_id]))
            donor_feature_cache[utt_id] = knn_vc.get_features(wav)
        return donor_feature_cache[utt_id]

    jobs = []
    for split, subset in FOLDERS:
        out_dir = vpc_data_dir / f"libri_{split}_{subset}{suffix}" / "wav"
        out_dir.mkdir(parents=True, exist_ok=True)
        for utt_id, true_speaker in folder_utterances(vpc_data_dir, split, subset):
            jobs.append((split, subset, utt_id, true_speaker, out_dir))

    if limit is not None:
        jobs = jobs[:limit]

    log.info(f"anonymizing {len(jobs)} utterances across {len(FOLDERS)} folders, suffix={suffix}")
    start = time.time()
    n_done, n_skipped, n_new = 0, 0, 0

    for split, subset, utt_id, true_speaker, out_dir in jobs:
        n_done += 1
        out_path = out_dir / f"{utt_id}.wav"
        if out_path.exists():
            n_skipped += 1
            continue

        donor_utts = tuple(sorted(assign_donors(utt_id, true_speaker, speaker_pool)))
        if donor_utts not in matching_set_cache:
            donor_feats = torch.cat([get_donor_features(u) for u in donor_utts], dim=0).cpu()
            matching_set_cache[donor_utts] = donor_feats
        matching_set = matching_set_cache[donor_utts]

        query_wav, sr = torchaudio.load(str(utt_path_index[utt_id]))
        query_seq = knn_vc.get_features(query_wav)
        out_wav = knn_vc.match(query_seq, matching_set, topk=k).cpu()
        torchaudio.save(str(out_path), out_wav[None], knn_vc.sr)
        n_new += 1

        if n_new % 25 == 0:
            elapsed = time.time() - start
            log.info(f"{n_done}/{len(jobs)} ({n_skipped} skipped) -- {eta_str(n_new, len(jobs) - n_done, elapsed)}")

    log.info(f"done: {n_new} newly anonymized, {n_skipped} already present, "
             f"{len(matching_set_cache)} distinct pseudo-speaker pools built")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vpc-root", default=str(DEFAULT_VPC_ROOT))
    parser.add_argument("--suffix", default="_knnvc")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--sanity", action="store_true", help="anonymize only the first 5 utterances")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    vpc_root = Path(args.vpc_root)
    vpc_data_dir = vpc_root / "data"
    setup_logging(vpc_root / "tier3_anonymize.log", verbose=args.verbose)
    log.info(f"args: {vars(args)}")

    device = get_device()
    log.info(f"device: {device}")

    utt_path_index = build_utt_path_index(vpc_data_dir)
    speaker_pool = load_speaker_pool(vpc_data_dir)
    log.info(f"raw pool: {len(utt_path_index)} utterances across {len(speaker_pool)} speakers")

    log.info("loading pretrained kNN-VC pipeline (WavLM + prematched HiFi-GAN)...")
    knn_vc = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=device)

    limit = 5 if args.sanity else None
    anonymize_all(knn_vc, vpc_data_dir, utt_path_index, speaker_pool, args.suffix, args.k, limit)


if __name__ == "__main__":
    main()
