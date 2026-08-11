"""Tier 3 step 1: reconstruct VPC2024's official LibriSpeech audio pool without the
password-gated SFTP download.

VPC2024's data.zip (public, no password: github.com/Voice-Privacy-Challenge/Voice-Privacy-
Challenge-2024/releases/download/data.zip/data.zip) ships the Kaldi metadata (wav.scp,
utt2spk, trials, spk2gender) for their official libri_dev/libri_test enroll+trial partition,
but not the audio itself -- wav.scp points at data/libri_{dev,test}/wav/<uttid>/<uttid>.wav,
which normally comes from a password-gated tarball. The utterance IDs are standard LibriSpeech
IDs (e.g. 1272-128104-0000), so the audio can be reconstructed from a plain public LibriSpeech
dev-clean/test-clean download instead. See docs/plan_tier_3.md.

Usage:
    python scripts/tier3_prepare_data.py --vpc-root third_party/Voice-Privacy-Challenge-2024
"""
import argparse
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import torchaudio

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VPC_ROOT = REPO_ROOT / "third_party" / "Voice-Privacy-Challenge-2024"
DEFAULT_LIBRISPEECH_ROOT = REPO_ROOT / "notebooks" / "data" / "LibriSpeech"
DATA_ZIP_URL = "https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/releases/download/data.zip/data.zip"

SPLIT_TO_LIBRISPEECH_DIR = {"dev": "dev-clean", "test": "test-clean"}
SPLIT_SUBSETS = ["enrolls", "trials_m", "trials_f"]


def ensure_metadata(vpc_root: Path) -> Path:
    """Download + unzip VPC's public data.zip if not already present."""
    data_dir = vpc_root / "data"
    marker = data_dir / "libri_dev_enrolls" / "wav.scp"
    if marker.exists():
        print(f"metadata already present: {data_dir}")
        return data_dir

    vpc_root.mkdir(parents=True, exist_ok=True)
    zip_path = vpc_root / "data.zip"
    if not zip_path.exists():
        print(f"downloading {DATA_ZIP_URL} -> {zip_path}")
        urlretrieve(DATA_ZIP_URL, zip_path)

    print(f"unzipping into {vpc_root}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(vpc_root)
    assert marker.exists(), f"expected {marker} after unzip, VPC's data.zip layout may have changed"
    return data_dir


def needed_utterance_ids(data_dir: Path, split: str) -> set[str]:
    utt_ids = set()
    for subset in SPLIT_SUBSETS:
        wav_scp = data_dir / f"libri_{split}_{subset}" / "wav.scp"
        for line in wav_scp.read_text().splitlines():
            utt_ids.add(line.split(maxsplit=1)[0])
    return utt_ids


def resolve_flac(librispeech_root: Path, split: str, utt_id: str) -> Path:
    speaker, chapter, _ = utt_id.split("-")
    return librispeech_root / SPLIT_TO_LIBRISPEECH_DIR[split] / speaker / chapter / f"{utt_id}.flac"


def reconstruct_pool(data_dir: Path, librispeech_root: Path, split: str, force: bool):
    utt_ids = needed_utterance_ids(data_dir, split)
    pool_dir = data_dir / f"libri_{split}" / "wav"
    pool_dir.mkdir(parents=True, exist_ok=True)

    missing, converted, skipped = [], 0, 0
    for i, utt_id in enumerate(sorted(utt_ids), 1):
        out_dir = pool_dir / utt_id
        out_path = out_dir / f"{utt_id}.wav"
        if out_path.exists() and not force:
            skipped += 1
            continue

        flac_path = resolve_flac(librispeech_root, split, utt_id)
        if not flac_path.exists():
            missing.append(utt_id)
            continue

        wav, sr = torchaudio.load(str(flac_path))
        out_dir.mkdir(exist_ok=True)
        torchaudio.save(str(out_path), wav, sr)
        converted += 1
        if i % 500 == 0:
            print(f"  {split}: {i}/{len(utt_ids)} processed")

    print(f"{split}: {converted} converted, {skipped} already present, {len(missing)} missing "
          f"(of {len(utt_ids)} needed)")
    if missing:
        print(f"  first few missing: {missing[:10]}")
    return converted, skipped, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vpc-root", default=str(DEFAULT_VPC_ROOT))
    parser.add_argument("--librispeech-root", default=str(DEFAULT_LIBRISPEECH_ROOT))
    parser.add_argument("--force", action="store_true", help="reconvert even if output wav already exists")
    args = parser.parse_args()

    vpc_root = Path(args.vpc_root)
    librispeech_root = Path(args.librispeech_root)

    data_dir = ensure_metadata(vpc_root)

    total_missing = []
    for split in ["dev", "test"]:
        _, _, missing = reconstruct_pool(data_dir, librispeech_root, split, args.force)
        total_missing.extend(missing)

    if total_missing:
        raise SystemExit(
            f"{len(total_missing)} utterances could not be resolved against "
            f"{librispeech_root} -- check dev-clean/test-clean are fully downloaded."
        )
    print("done: data/libri_dev/wav and data/libri_test/wav are ready for anonymization input.")


if __name__ == "__main__":
    main()
