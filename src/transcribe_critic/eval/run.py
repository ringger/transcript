"""Orchestrate transcribe-critic pipeline runs over dataset manifest entries."""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from transcribe_critic.shared import TRANSCRIPT_MD, AUDIO_MP3, METADATA_JSON
from transcribe_critic.eval.datasets import load_manifest, filter_manifest


def _is_complete(output_dir: Path) -> bool:
    """Check if a pipeline run completed for this file.

    A run is considered complete if transcript.md exists (the final stage output).
    """
    return (output_dir / TRANSCRIPT_MD).exists()


def _seed_output_dir(output_dir: Path, audio_path: Path, entry) -> None:
    """Seed the output directory so the pipeline skips the download stage.

    The pipeline expects audio.mp3 and metadata.json in the output dir.
    If those already exist, the download stage is skipped via skip_existing.
    We symlink the local audio file and create a minimal metadata.json.
    """
    # Symlink audio as audio.mp3 (Whisper handles wav/mp3/flac regardless of extension)
    audio_link = output_dir / AUDIO_MP3
    if not audio_link.exists():
        abs_audio = audio_path.resolve()
        os.symlink(abs_audio, audio_link)

    # Create minimal metadata.json
    metadata_path = output_dir / METADATA_JSON
    if not metadata_path.exists():
        metadata = {
            "url": f"local://{audio_path}",
            "video_id": entry.file_id,
            "title": entry.metadata.get("episode_title")
                     or entry.metadata.get("company_name")
                     or entry.file_id,
            "channel": entry.metadata.get("show_title", ""),
            "upload_date": None,
            "duration_seconds": entry.duration_secs,
            "description": "",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))


def run_pipeline(args):
    """Entry point for the run subcommand."""
    manifest_path = Path(args.manifest)
    dataset_dir = manifest_path.parent

    file_ids = args.file_ids.split(",") if args.file_ids else None

    entries = load_manifest(manifest_path)
    entries = filter_manifest(
        entries,
        subset=args.subset,
        max_files=args.max_files,
        max_hours=args.max_hours,
        file_ids=file_ids,
    )

    if not entries:
        print("No files to process after filtering.", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(f"./eval-runs/{timestamp}")

    run_dir.mkdir(parents=True, exist_ok=True)

    total_hours = sum(e.duration_secs for e in entries) / 3600
    print(f"Processing {len(entries)} files ({total_hours:.1f} hours)")
    print(f"Output: {run_dir}")
    print()

    # Track status
    status_path = run_dir / "run_status.json"
    status = {}
    if status_path.exists():
        status = json.loads(status_path.read_text())

    # Use a dummy URL — the pipeline will skip download because audio.mp3 already exists
    dummy_url = "https://example.com/eval-placeholder"

    if args.dry_run:
        for entry in entries:
            output_dir = run_dir / f"file{entry.file_id}"
            complete = _is_complete(output_dir) and not args.force
            state = "skip (complete)" if complete else "would run"
            print(f"  {entry.file_id}: {state}")
            if not complete:
                cmd = ["transcribe-critic", dummy_url, "--podcast", "--no-slides",
                       "-o", str(output_dir)]
                print(f"    {' '.join(cmd)}")
        return

    succeeded = 0
    failed = 0
    skipped = 0

    for i, entry in enumerate(entries, 1):
        output_dir = run_dir / f"file{entry.file_id}"
        audio_path = dataset_dir / entry.audio_path

        # Skip if already complete
        if _is_complete(output_dir) and not args.force:
            if args.verbose:
                print(f"  [{i}/{len(entries)}] {entry.file_id}: skipping (complete)")
            skipped += 1
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        # Seed the output dir with audio symlink + metadata so download is skipped
        _seed_output_dir(output_dir, audio_path, entry)

        cmd = [
            "transcribe-critic",
            dummy_url,
            "--podcast",
            "--no-slides",
            "-o", str(output_dir),
        ]

        print(f"  [{i}/{len(entries)}] {entry.file_id} ({_fmt_duration(entry.duration_secs)})...")
        if args.verbose:
            print(f"    {' '.join(cmd)}")

        # Redirect output to log file
        log_path = output_dir / "pipeline.log"
        try:
            with open(log_path, "w") as log_file:
                result = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=7200,  # 2 hour timeout per file
                )

            if result.returncode == 0:
                status[entry.file_id] = {"status": "success", "timestamp": datetime.now().isoformat()}
                succeeded += 1
                print(f"    done")
            else:
                status[entry.file_id] = {"status": "failed", "returncode": result.returncode}
                failed += 1
                print(f"    FAILED (exit code {result.returncode}, see {log_path})")

        except subprocess.TimeoutExpired:
            status[entry.file_id] = {"status": "timeout"}
            failed += 1
            print(f"    TIMEOUT (see {log_path})")
        except Exception as e:
            status[entry.file_id] = {"status": "error", "message": str(e)}
            failed += 1
            print(f"    ERROR: {e}")

        # Save status after each file
        status_path.write_text(json.dumps(status, indent=2))

    print(f"\nDone: {succeeded} succeeded, {failed} failed, {skipped} skipped")
    print(f"Results in: {run_dir}")


def _fmt_duration(secs: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
