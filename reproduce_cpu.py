"""Regenerate and byte-compare every committed CPU-derived artifact.

This intentionally excludes sealed outputs whose raw activation/transcript inputs
are off-git. It never invokes Modal, a model provider, a network API, or a GPU.
Run from any directory with:

    uv run --frozen python reproduce_cpu.py
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

RUN_V1_DATA = ROOT / "experiments/exp3_dataset/data"
LEAK_FREE_DATA = ROOT / "experiments/exp3_dataset/data_leak_free"
RUN_V1_BASELINE = (
    ROOT / "experiments/exp4_analysis/results/wave4/full_information_text_baseline.json"
)
EXP4B_BASELINE = (
    ROOT / "experiments/exp4_analysis/results/exp4b/full_information_text_baseline.json"
)
HEADLINE_FIGURE = ROOT / "docs/writeup/assets/wave4-headline-results.svg"
EXP4B_FIGURE = ROOT / "docs/writeup/assets/exp4b-leak-free-text-baseline.svg"


def _run(*args: str) -> None:
    subprocess.run([PYTHON, *args], cwd=ROOT, check=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare_file(generated: Path, committed: Path) -> None:
    if generated.read_bytes() != committed.read_bytes():
        raise RuntimeError(f"byte mismatch: {committed.relative_to(ROOT)}")
    print(f"PASS  {_sha256(committed)}  {committed.relative_to(ROOT)}")


def _compare_directory(
    generated: Path, committed: Path, *, excluded_committed: frozenset[Path] = frozenset()
) -> None:
    generated_files = sorted(
        path.relative_to(generated) for path in generated.rglob("*") if path.is_file()
    )
    committed_files = sorted(
        relative
        for path in committed.rglob("*")
        if path.is_file()
        and (relative := path.relative_to(committed)) not in excluded_committed
    )
    if generated_files != committed_files:
        raise RuntimeError(
            f"file inventory mismatch: {committed.relative_to(ROOT)} "
            f"(generated={generated_files}, committed={committed_files})"
        )
    for relative in committed_files:
        _compare_file(generated / relative, committed / relative)
    for relative in sorted(excluded_committed):
        if not (committed / relative).is_file():
            raise RuntimeError(f"declared exclusion is missing: {committed / relative}")
        print(f"SKIP  GPU-derived artifact  {(committed / relative).relative_to(ROOT)}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="vmp-cpu-reproduce-") as raw_tmp:
        tmp = Path(raw_tmp)
        generated_run_v1 = tmp / "data"
        generated_leak_free = tmp / "data_leak_free"
        generated_run_v1_baseline = tmp / "full_information_text_baseline.json"
        generated_exp4b_baseline = tmp / "exp4b_full_information_text_baseline.json"
        generated_headline = tmp / "wave4-headline-results.svg"
        generated_exp4b_figure = tmp / "exp4b-leak-free-text-baseline.svg"

        _run(
            "-m",
            "experiments.exp3_dataset.build_dataset",
            "--output-dir",
            str(generated_run_v1),
        )
        _compare_directory(
            generated_run_v1,
            RUN_V1_DATA,
            excluded_committed=frozenset({Path("sanity_results.json")}),
        )

        _run(
            "-m",
            "experiments.exp3_dataset.build_dataset",
            "--leak-free",
            "--output-dir",
            str(generated_leak_free),
        )
        _compare_directory(generated_leak_free, LEAK_FREE_DATA)

        _run(
            "-m",
            "experiments.exp4_analysis.full_information_text_baseline",
            "--output",
            str(generated_run_v1_baseline),
        )
        _compare_file(generated_run_v1_baseline, RUN_V1_BASELINE)

        _run(
            "-m",
            "experiments.exp4_analysis.full_information_text_baseline",
            "--leak-free",
            "--output",
            str(generated_exp4b_baseline),
        )
        _compare_file(generated_exp4b_baseline, EXP4B_BASELINE)

        _run(
            "-m",
            "experiments.exp4_analysis.generate_headline_figure",
            "--baseline",
            str(generated_run_v1_baseline),
            "--output",
            str(generated_headline),
        )
        _compare_file(generated_headline, HEADLINE_FIGURE)

        _run(
            "-m",
            "experiments.exp4_analysis.generate_exp4b_text_baseline_figure",
            "--original",
            str(generated_run_v1_baseline),
            "--leak-free",
            str(generated_exp4b_baseline),
            "--output",
            str(generated_exp4b_figure),
        )
        _compare_file(generated_exp4b_figure, EXP4B_FIGURE)

    print("PASS: every committed CPU-derived artifact is byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
