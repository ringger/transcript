"""Dataset download, manifest generation, and subset filtering.

Supported datasets:
- Earnings-21: Multi-speaker earnings calls with professional transcripts + RTTM
- Rev16: Podcast episodes with professional transcripts (no speaker labels)
"""

import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ManifestEntry:
    """A single evaluation file with paths to audio and reference files."""
    file_id: str
    audio_path: str          # Relative to manifest directory
    ref_stm_path: str        # Reference STM (pre-converted)
    ref_rttm_path: Optional[str] = None  # Reference RTTM (Earnings-21 only)
    duration_secs: float = 0.0
    subset_tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def save_manifest(entries: list[ManifestEntry], manifest_path: Path):
    """Save manifest to JSON."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "entries": [asdict(e) for e in entries],
    }
    manifest_path.write_text(json.dumps(data, indent=2))


def load_manifest(manifest_path: Path) -> list[ManifestEntry]:
    """Load manifest from JSON."""
    data = json.loads(manifest_path.read_text())
    return [ManifestEntry(**e) for e in data["entries"]]


def filter_manifest(
    entries: list[ManifestEntry],
    subset: Optional[str] = None,
    max_files: Optional[int] = None,
    max_hours: Optional[float] = None,
    file_ids: Optional[list[str]] = None,
) -> list[ManifestEntry]:
    """Filter manifest entries by subset, count, duration, or explicit IDs."""
    filtered = entries

    if subset:
        filtered = [e for e in filtered if subset in e.subset_tags]

    if file_ids:
        id_set = set(file_ids)
        filtered = [e for e in filtered if e.file_id in id_set]

    if max_hours is not None:
        result = []
        total = 0.0
        for e in filtered:
            if total + e.duration_secs / 3600 > max_hours:
                break
            result.append(e)
            total += e.duration_secs / 3600
        filtered = result

    if max_files is not None:
        filtered = filtered[:max_files]

    return filtered


# --- Earnings-21 ---

EARNINGS21_REPO = "https://github.com/revdotcom/speech-datasets.git"
EARNINGS21_EVAL10_IDS = {
    "4320211", "4341191", "4346818", "4359971", "4365024",
    "4366522", "4366893", "4367535", "4383161", "4384964", "4387332",
}


def _prep_earnings21(data_dir: Path, subset: Optional[str]):
    """Download and prepare Earnings-21 dataset."""
    dataset_dir = data_dir / "earnings21"
    repo_dir = dataset_dir / "repo"

    # Clone the repo (sparse checkout for just earnings21)
    if not (repo_dir / ".git").exists():
        print(f"Cloning speech-datasets repo to {repo_dir}...")
        repo_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--sparse",
             EARNINGS21_REPO, str(repo_dir)],
            check=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "earnings21"],
            cwd=repo_dir, check=True,
        )
        print("Pulling audio files with Git LFS...")
        subprocess.run(["git", "lfs", "pull"], cwd=repo_dir, check=True)
    else:
        print(f"Repository already exists at {repo_dir}")

    e21_dir = repo_dir / "earnings21"

    # Load metadata
    metadata_csv = e21_dir / "earnings21-file-metadata.csv"
    if not metadata_csv.exists():
        print(f"Error: metadata CSV not found at {metadata_csv}", file=sys.stderr)
        sys.exit(1)

    file_metadata = {}
    with open(metadata_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = row["file_id"]
            file_metadata[fid] = {
                "company_name": row.get("company_name", ""),
                "audio_length": float(row.get("audio_length", 0)),
                "unique_speakers": int(row.get("unique_speakers", 0)),
                "sector": row.get("sector", ""),
            }

    # Convert references to STM and normalize RTTM
    from transcribe_critic.eval.convert import nlp_to_stm

    ref_dir = dataset_dir / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)

    nlp_dir = e21_dir / "transcripts" / "nlp_references"
    rttm_dir = e21_dir / "rttms"
    media_dir = e21_dir / "media"

    entries = []
    nlp_files = sorted(nlp_dir.glob("*.nlp")) if nlp_dir.exists() else []

    if not nlp_files:
        print(f"Error: No NLP files found in {nlp_dir}", file=sys.stderr)
        sys.exit(1)

    for nlp_path in nlp_files:
        file_id = nlp_path.stem
        meta = file_metadata.get(file_id, {})

        # Determine subset tags
        tags = ["all"]
        if file_id in EARNINGS21_EVAL10_IDS:
            tags.append("eval10")

        # Skip if subset filter is active and this file isn't in it
        if subset and subset not in tags:
            continue

        # Find audio file
        audio_path = None
        for ext in (".mp3", ".wav", ".flac"):
            candidate = media_dir / f"{file_id}{ext}"
            if candidate.exists():
                audio_path = candidate
                break

        if audio_path is None:
            print(f"  Warning: No audio file for {file_id}, skipping")
            continue

        # Convert NLP to STM
        stm_path = ref_dir / f"{file_id}.stm"
        stm_content = nlp_to_stm(nlp_path, file_id)
        stm_path.write_text(stm_content)

        # Copy RTTM if available
        rttm_src = rttm_dir / f"{file_id}.rttm"
        rttm_dst = ref_dir / f"{file_id}.rttm"
        if rttm_src.exists():
            shutil.copy2(rttm_src, rttm_dst)

        entry = ManifestEntry(
            file_id=file_id,
            audio_path=str(audio_path.relative_to(dataset_dir)),
            ref_stm_path=str(stm_path.relative_to(dataset_dir)),
            ref_rttm_path=str(rttm_dst.relative_to(dataset_dir)) if rttm_dst.exists() else None,
            duration_secs=meta.get("audio_length", 0),
            subset_tags=tags,
            metadata=meta,
        )
        entries.append(entry)

    manifest_path = dataset_dir / "manifest.json"
    save_manifest(entries, manifest_path)

    n_eval10 = sum(1 for e in entries if "eval10" in e.subset_tags)
    total_hours = sum(e.duration_secs for e in entries) / 3600
    print(f"\nPrepared {len(entries)} files ({total_hours:.1f} hours)")
    print(f"  eval10 subset: {n_eval10} files")
    print(f"  Manifest: {manifest_path}")


# --- Rev16 ---

def _prep_rev16(data_dir: Path, subset: Optional[str]):
    """Download and prepare Rev16 dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "Error: 'datasets' package required for Rev16. "
            "Install with: pip install transcribe-critic[eval]",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset_dir = data_dir / "rev16"
    audio_dir = dataset_dir / "audio"
    ref_dir = dataset_dir / "references"
    audio_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    config_name = "whisper_subset" if subset == "whisper_subset" else "full"
    print(f"Loading Rev16 ({config_name}) from HuggingFace...")
    ds = load_dataset("distil-whisper/rev16", config_name, split="test")

    from transcribe_critic.eval.convert import plain_text_to_stm

    entries = []
    for i, row in enumerate(ds):
        file_id = row.get("file_number", str(i))

        # Save audio to WAV
        audio_data = row["audio"]
        wav_path = audio_dir / f"{file_id}.wav"
        if not wav_path.exists():
            import soundfile as sf
            sf.write(str(wav_path), audio_data["array"], audio_data["sampling_rate"])

        # Get duration
        duration = len(audio_data["array"]) / audio_data["sampling_rate"]

        # Convert reference text to STM
        ref_text = row.get("transcription", "")
        stm_path = ref_dir / f"{file_id}.stm"
        # Write text to temp file and convert
        tmp_txt = ref_dir / f"{file_id}_ref.txt"
        tmp_txt.write_text(ref_text)
        stm_content = plain_text_to_stm(tmp_txt, file_id)
        stm_path.write_text(stm_content)
        tmp_txt.unlink()

        tags = ["all"]
        if config_name == "whisper_subset":
            tags.append("whisper_subset")

        entry = ManifestEntry(
            file_id=file_id,
            audio_path=str(wav_path.relative_to(dataset_dir)),
            ref_stm_path=str(stm_path.relative_to(dataset_dir)),
            ref_rttm_path=None,  # Rev16 has no speaker labels
            duration_secs=duration,
            subset_tags=tags,
            metadata={
                "show_title": row.get("show_title", ""),
                "episode_title": row.get("episode_title", ""),
            },
        )
        entries.append(entry)

    manifest_path = dataset_dir / "manifest.json"
    save_manifest(entries, manifest_path)

    total_hours = sum(e.duration_secs for e in entries) / 3600
    print(f"\nPrepared {len(entries)} files ({total_hours:.1f} hours)")
    print(f"  Manifest: {manifest_path}")


def prep_dataset(args):
    """Entry point for the prep subcommand."""
    data_dir = Path(args.data_dir)

    if args.dataset == "earnings21":
        _prep_earnings21(data_dir, args.subset)
    elif args.dataset == "rev16":
        _prep_rev16(data_dir, args.subset)
    else:
        print(f"Unknown dataset: {args.dataset}", file=sys.stderr)
        sys.exit(1)
