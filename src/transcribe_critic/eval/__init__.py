"""Evaluation harness for transcribe-critic.

Subcommands:
    prep   — Download dataset, convert references, write manifest
    run    — Run transcribe-critic pipeline on manifest entries
    score  — Score hypothesis outputs against references using meeteval
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="transcribe-critic-eval",
        description="Evaluate transcribe-critic against speech benchmarks",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- prep ---
    prep_parser = subparsers.add_parser(
        "prep", help="Download dataset and prepare references"
    )
    prep_parser.add_argument(
        "dataset",
        choices=["earnings21", "rev16"],
        help="Dataset to prepare",
    )
    prep_parser.add_argument(
        "--data-dir",
        type=str,
        default="./eval-data",
        help="Where to store downloaded data (default: ./eval-data)",
    )
    prep_parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Subset to prepare: eval10 (earnings21), whisper_subset (rev16)",
    )

    # --- run ---
    run_parser = subparsers.add_parser(
        "run", help="Run transcribe-critic pipeline on dataset files"
    )
    run_parser.add_argument("manifest", help="Path to manifest.json from prep phase")
    run_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to store pipeline outputs (default: auto-generated)",
    )
    run_parser.add_argument("--subset", type=str, default=None)
    run_parser.add_argument("--max-files", type=int, default=None)
    run_parser.add_argument("--max-hours", type=float, default=None)
    run_parser.add_argument(
        "--file-ids", type=str, default=None,
        help="Comma-separated file IDs to process",
    )
    run_parser.add_argument(
        "--pipeline-args",
        type=str,
        default="",
        help="Additional args passed to transcribe-critic (quoted string)",
    )
    run_parser.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    run_parser.add_argument("--dry-run", action="store_true", help="Show what would be run")
    run_parser.add_argument("-v", "--verbose", action="store_true")

    # --- score ---
    score_parser = subparsers.add_parser(
        "score", help="Score pipeline outputs against references"
    )
    score_parser.add_argument("manifest", help="Path to manifest.json")
    score_parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Directory containing pipeline outputs",
    )
    score_parser.add_argument("--subset", type=str, default=None)
    score_parser.add_argument("--max-files", type=int, default=None)
    score_parser.add_argument(
        "--file-ids", type=str, default=None,
        help="Comma-separated file IDs to score",
    )
    score_parser.add_argument(
        "--metrics",
        type=str,
        default="wer",
        help="Comma-separated metrics: wer,cpwer,der (default: wer)",
    )
    score_parser.add_argument(
        "--hypothesis",
        type=str,
        default="all",
        choices=["all", "auto", "merged", "diarized", "whisper"],
        help="Which output(s) to evaluate: 'all' scores every variant (default: all)",
    )
    score_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write results to file (markdown). Default: stdout + {run-dir}/results.md",
    )

    args = parser.parse_args()

    if args.command == "prep":
        from transcribe_critic.eval.datasets import prep_dataset
        prep_dataset(args)
    elif args.command == "run":
        from transcribe_critic.eval.run import run_pipeline
        run_pipeline(args)
    elif args.command == "score":
        from transcribe_critic.eval.score import score_results
        score_results(args)


if __name__ == "__main__":
    main()
