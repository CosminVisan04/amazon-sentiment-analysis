"""
run_all.py — end-to-end pipeline runner.
Runs all phases in order; each phase is cache-aware and skips if outputs exist.
Usage: python run_all.py
       python run_all.py --force   # ignore all caches
"""
import argparse
import subprocess
import sys
from pathlib import Path

PHASES = [
    "src/01_sample.py",
    "src/02_preprocess.py",
    "src/03_sentiment.py",
    "src/04_rq1_distributions.py",
    "src/05_rq2_temporal.py",
    "src/06_rq3_agreement.py",
    "src/07_rq4_words.py",
]

ROOT = Path(__file__).parent


def run_phase(script: str, force: bool):
    path = ROOT / script
    if not path.exists():
        print(f"[SKIP] {script} — not yet implemented")
        return
    env_args = ["--force"] if force else []
    result = subprocess.run(
        [sys.executable, str(path)] + env_args,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-run all phases ignoring cache")
    args = parser.parse_args()

    for phase in PHASES:
        print(f"\n{'='*60}")
        print(f"  Running {phase}")
        print('='*60)
        run_phase(phase, args.force)

    print("\nAll phases complete.")


if __name__ == "__main__":
    main()
