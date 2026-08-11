"""Tier 3 step 2: pseudo-speaker assignment for kNN-VC anonymization.

kNN-VC as published always converts to one fixed real target speaker -- fine for voice
conversion, meaningless for anonymization (it just swaps whose real identity leaks). This
builds, for each utterance, a pooled matching set drawn from multiple donor speakers instead
of one real target, so no single output utterance's k-NN converted frames trace back to one
identifiable person. See docs/plan_tier_3.md for the design rationale.

Assignment is utterance-level (each utterance gets its own donor pool, not one fixed pool per
real speaker) and deterministic (hash of the utterance ID seeds donor selection), so re-running
this against the same data always produces the same pseudo-speaker assignment.

Pure data-structure logic, no torch/audio I/O -- kept separate from tier3_anonymize.py so the
assignment scheme itself is directly testable.
"""
import hashlib
import random
from pathlib import Path

N_DONORS = 8
UTTS_PER_DONOR = 3


def load_speaker_pool(vpc_data_dir: Path, splits=("dev", "test")) -> dict[str, list[str]]:
    """speaker_id -> sorted list of that speaker's utterance IDs, from the reconstructed
    raw wav pools (data/libri_dev/wav, data/libri_test/wav)."""
    pool: dict[str, list[str]] = {}
    for split in splits:
        wav_root = vpc_data_dir / f"libri_{split}" / "wav"
        if not wav_root.exists():
            continue
        for utt_dir in wav_root.iterdir():
            utt_id = utt_dir.name
            speaker_id = utt_id.split("-")[0]
            pool.setdefault(speaker_id, []).append(utt_id)
    for utts in pool.values():
        utts.sort()
    return pool


def assign_donors(utt_id: str, true_speaker: str, speaker_pool: dict[str, list[str]],
                   n_donors: int = N_DONORS, utts_per_donor: int = UTTS_PER_DONOR) -> list[str]:
    """Deterministic donor-utterance list for one query utterance. Never draws from the
    utterance's own real speaker. Returns a flat list of donor utterance IDs (the matching-set
    material), same length every call for the same utt_id."""
    candidates = sorted(spk for spk in speaker_pool if spk != true_speaker)
    if len(candidates) < n_donors:
        raise ValueError(f"only {len(candidates)} candidate donor speakers, need {n_donors}")

    rng = random.Random(int(hashlib.sha256(utt_id.encode()).hexdigest(), 16))
    donor_speakers = rng.sample(candidates, n_donors)

    donor_utts = []
    for spk in donor_speakers:
        available = speaker_pool[spk]
        k = min(utts_per_donor, len(available))
        donor_utts.extend(rng.sample(available, k))
    return donor_utts
