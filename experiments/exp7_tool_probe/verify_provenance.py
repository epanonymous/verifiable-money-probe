"""Check (or write) the Exp 7 provenance manifest ``data/<ver>/shards.sha256``.

    uv run python -m experiments.exp7_tool_probe.verify_provenance --acts experiments/exp7_tool_probe/local/v0
    uv run python -m experiments.exp7_tool_probe.verify_provenance --acts <dir> --write            # regenerate
    uv run python -m experiments.exp7_tool_probe.verify_provenance --report-only                   # lenient (CI, no shards)
    uv run python -m experiments.exp7_tool_probe.verify_provenance --run-version v1 --acts <dir>   # another run

Entries under ``acts/`` are the off-git activation shards (Modal volume
``vmp-activations``, path ``exp7/<ver>``; local mirror ``local/<ver>``).

Fails closed. Exit status: 0 only when every listed entry is present and
matches; 1 on any mismatch; 2 on anything missing (including all 168 shards
when ``--acts`` is not given). ``--report-only`` restores the old lenient
behaviour (missing entries are named but exit 0) for environments that cannot
hold the shards, such as CI; it is never the advertised verification command.
The expected shard inventory (48 main + 120 auth for v0) is read from
``data/<ver>/manifest.json``; a manifest that lists fewer shards fails too.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import provenance as prov

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_RUN_VERSION = "v0"
DATA = HERE / "data" / DEFAULT_RUN_VERSION
RESULTS = HERE / "results" / DEFAULT_RUN_VERSION
MANIFEST = DATA / prov.SHA_MANIFEST_NAME
FALLBACK_EXPECTED_SHARDS = {"main": 48, "auth": 120}
EXPECTED_SHARDS = FALLBACK_EXPECTED_SHARDS  # v0 inventory; ``expected_shards()`` reads the data manifest


def run_paths(run_version: str) -> dict[str, Path]:
    data = HERE / "data" / run_version
    return {"data": data, "results": HERE / "results" / run_version, "manifest": data / prov.SHA_MANIFEST_NAME}


def expected_shards(data_dir: Path) -> dict[str, int]:
    """48 main + 120 auth for v0, read from the data manifest when it is there."""

    manifest = data_dir / "manifest.json"
    if manifest.is_file():
        m = json.loads(manifest.read_text())
        if "n_templates" in m and "n_auth_rows" in m:
            return {"main": int(m["n_templates"]), "auth": int(m["n_auth_rows"])}
    return dict(FALLBACK_EXPECTED_SHARDS)


def build_records(acts_dir: Path | None, *, data: Path = DATA, results: Path = RESULTS) -> list[dict]:
    records = prov.collect_records(data, prefix=data.relative_to(REPO).as_posix(), exclude={prov.SHA_MANIFEST_NAME})
    if results.is_dir():
        records += prov.collect_records(results, prefix=results.relative_to(REPO).as_posix())
    if acts_dir is not None:
        for sub in ("main", "auth"):
            records += prov.collect_records(acts_dir / sub, prefix=f"{prov.ACTS_PREFIX}/{sub}")
    return sorted(records, key=lambda r: r["path"])


def write_manifest(acts_dir: Path | None, *, allow_partial: bool, run_version: str = DEFAULT_RUN_VERSION) -> int:
    paths = run_paths(run_version)
    expected = expected_shards(paths["data"])
    records = build_records(acts_dir, data=paths["data"], results=paths["results"])
    shards = {sub: [r for r in records if r["path"].startswith(f"{prov.ACTS_PREFIX}/{sub}/")] for sub in expected}
    for sub, n in expected.items():
        if len(shards[sub]) != n and not allow_partial:
            raise SystemExit(f"{sub}: {len(shards[sub])} shards present, {n} expected; pass --allow-partial to write anyway")
    acts_records = [r for r in records if r["path"].startswith(prov.ACTS_PREFIX + "/")]
    header = [
        f"Exp 7 {run_version} provenance manifest: sha256  bytes  path",
        f"acts/ entries are off-git activation shards (Modal volume vmp-activations, exp7/{run_version};",
        f"local mirror experiments/exp7_tool_probe/local/{run_version}). Verify (fails closed) with:",
        f"  uv run python -m experiments.exp7_tool_probe.verify_provenance --run-version {run_version} --acts <dir>",
        f"shards: main={len(shards['main'])} auth={len(shards['auth'])}; aggregate sha256 over acts/ = {prov.aggregate_sha256(acts_records)}",
    ]
    prov.write_sha256_manifest(paths["manifest"], records, header)
    print(f"wrote {paths['manifest'].relative_to(REPO)}: {len(records)} entries ({len(acts_records)} shards)")
    return 0


def _exit_code(report: dict[str, list[dict]], inventory_problems: list[str], *, report_only: bool) -> int:
    if report["mismatch"]:
        return 1
    if report_only:
        return 0
    if report["missing"] or inventory_problems:
        return 2
    return 0


def inventory_problems(records: list[dict], expected: dict[str, int]) -> list[str]:
    problems = []
    for sub, n in expected.items():
        listed = sum(1 for r in records if r["path"].startswith(f"{prov.ACTS_PREFIX}/{sub}/"))
        if listed != n:
            problems.append(f"manifest lists {listed} {sub} shards, {n} expected")
    return problems


def verify(acts_dir: Path | None, *, report_only: bool, run_version: str = DEFAULT_RUN_VERSION, as_json: bool = False) -> int:
    paths = run_paths(run_version)
    if not paths["manifest"].is_file():
        print(f"no manifest at {paths['manifest'].relative_to(REPO)}", file=sys.stderr)
        return 2
    records = prov.read_sha256_manifest(paths["manifest"])
    expected = expected_shards(paths["data"])
    problems = inventory_problems(records, expected)
    report = prov.verify_records(records, repo_root=REPO, acts_dir=acts_dir)
    code = _exit_code(report, problems, report_only=report_only)
    if as_json:
        print(
            json.dumps(
                {k: len(v) for k, v in report.items()}
                | {
                    "missing_paths": [r["path"] for r in report["missing"]],
                    "mismatch_paths": [r["path"] for r in report["mismatch"]],
                    "inventory_problems": problems,
                    "strict": not report_only,
                    "exit_code": code,
                },
                indent=2,
            )
        )
        return code
    data_prefix = paths["data"].relative_to(REPO).as_posix() + "/"
    results_prefix = paths["results"].relative_to(REPO).as_posix() + "/"
    groups = {f"data/{run_version}": data_prefix, f"results/{run_version}": results_prefix, "acts/main": "acts/main/", "acts/auth": "acts/auth/"}
    for name, prefix in groups.items():
        counts = {k: sum(1 for r in v if r["path"].startswith(prefix)) for k, v in report.items()}
        print(f"{name:<11} ok={counts['ok']:<4} mismatch={counts['mismatch']:<4} missing={counts['missing']}")
    for rec in report["mismatch"]:
        print(f"MISMATCH {rec['path']}: manifest {rec['sha256'][:12]}/{rec['bytes']}B, local {rec['local_sha256'][:12]}/{rec['local_bytes']}B")
    for rec in report["missing"]:
        print(f"MISSING  {rec['path']} ({rec['bytes']} bytes, sha256 {rec['sha256'][:12]}...)")
    for problem in problems:
        print(f"INVENTORY {problem}")
    acts_present = [r for r in report["ok"] + report["mismatch"] if r["path"].startswith(prov.ACTS_PREFIX + "/")]
    if acts_dir is None:
        print("note: no --acts given; the off-git shards are MISSING (exit 2 unless --report-only)")
    elif acts_present:
        print(f"acts aggregate sha256 (present shards only) = {prov.aggregate_sha256([{k: r[k] for k in ('path', 'sha256', 'bytes')} for r in acts_present])}")
    print({0: "OK: every listed entry present and matching", 1: "FAIL: mismatch", 2: "FAIL: missing entries or incomplete inventory"}[code] + ("" if not report_only else " (report-only: missing entries do not fail)"))
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-version", default=DEFAULT_RUN_VERSION, help="which data/<ver> + results/<ver> to check (default v0)")
    parser.add_argument("--acts", default=None, help="dir holding main/ and auth/ shards (off-git)")
    parser.add_argument("--write", action="store_true", help="regenerate the manifest from what is present")
    parser.add_argument("--allow-partial", action="store_true", help="with --write: accept fewer than the expected shards")
    parser.add_argument("--report-only", action="store_true", help="lenient: name missing entries but exit 0 (mismatches still exit 1)")
    parser.add_argument("--strict", action="store_true", help="(default now; kept for old scripts) exit 2 if anything listed is missing")
    parser.add_argument("--json", action="store_true", help="print the verification report as JSON")
    args = parser.parse_args(argv)
    if args.strict and args.report_only:
        parser.error("--strict and --report-only contradict each other")
    acts_dir = Path(args.acts) if args.acts else None
    if args.write:
        return write_manifest(acts_dir, allow_partial=args.allow_partial, run_version=args.run_version)
    return verify(acts_dir, report_only=args.report_only, run_version=args.run_version, as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
