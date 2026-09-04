# Exp 7 v0 — single-turn tool-grounded belief probe

Design lock: internal tracker issue #31 (private, omitted).
(v0 scope; the multi-turn live loop is a separate, unlocked pre-registration).

One model (Qwen3-30B-A3B-Instruct-2507) sees a byte-identical "you've been paid"
context in two worlds, makes ONE `check_balance` tool call itself, and gets a live
Base-mainnet USDC readout that differs only because the worlds differ: REAL is bound
server-side to a wallet that held 1.90 USDC at Base block 50836993, SHAM to a wallet
that never received funds. Activations are probed at P0 (before the tool result;
must be at chance) and P1 (after; before the SPEND/HOLD decision). A side probe (V5)
asks whether real Base JSON-RPC responses are distinguishable from well-forged
fakes as text. No custody, signing, or spending anywhere in v0; the RPC client is
allowlisted read-only.

Results: [`results/v0/summary.md`](results/v0/summary.md) (numbers, caveats,
pre-registration deviations, provenance), `results/v0/results.json`,
`results/v0/blind_p0.json` (+ `llm_judge_record.json`, the independent judge's raw
replies), figures, `results/v0/transcripts.jsonl`.

## Layout

| file | role |
|---|---|
| `config.py` | the single `SEED`, every locked constant, `config_hash`, `derive_seed`, `seed_everything` |
| `templates.py`, `dataset.py` | paraphrase bank, 48 templates (38 train / 10 held-out), REAL/SHAM rows |
| `worlds.py`, `rpc.py` | the one honest tool; block-pinned drift guard; read-only JSON-RPC client |
| `context.py`, `collect.py` | context assembly, strict model-emitted tool-call check, pair-level P0 identity |
| `authenticity.py` | V5 captures, forgeries, stratified-by-kind pair split |
| `prepare.py` | freezes `data/<ver>/` (needs Base RPC) |
| `modal_collect.py` | GPU collector (Modal): every path derived from `EXP7_RUN_VERSION`, immutable model revision required, full collection identity stamped into each shard, resume refuses any mismatch |
| `analysis.py` | CPU analysis of the shards → `results/<ver>/`; preflights the shard inventory against `shards.sha256` first |
| `blind_p0.py` | blind P0 gate: deterministic text audit (TF-IDF judge, lexicon, shuffle null, P1 control) **and** the locked independent-model judge (`--llm`; raw record + offline replay); verdict is INCOMPLETE without the model unless `--allow-no-llm` |
| `provenance.py`, `verify_provenance.py` | sha256 manifests for shards and artifacts; collection identity; verifier that fails closed |
| `data/v0/` | frozen inputs, `auth_split.json`, `manifest.json` (collection record + amendments), `shards.sha256` |
| `tests/` | CPU tests (gates, split, blind gate incl. judge plumbing, analysis on synthetic shards, provenance/identity/preflight) |

## Reproduce the analysis (CPU only)

The raw activations (48 main + 120 auth `.npz` shards, ~190 MB) are off-git on Modal
volume `vmp-activations` under `exp7/v0` (a gitignored local mirror lives at
`experiments/exp7_tool_probe/local/v0`, gitignored). Everything else is in the repo.

```bash
uv sync --frozen                                     # numpy/scikit-learn/scipy/matplotlib/modal pinned in uv.lock

# 1. fetch the shards (Modal access needed; nothing is launched)
MODAL_PROFILE=<your-modal-profile> modal volume get vmp-activations exp7/v0 experiments/exp7_tool_probe/local/v0

# 2. check every input, result and shard against the committed manifest. Fails closed:
#    exit 2 if anything listed is missing (e.g. no --acts -> all 168 shards), 1 if anything differs.
#    --report-only is the lenient mode for shard-less environments such as CI and is not a verification.
uv run python -m experiments.exp7_tool_probe.verify_provenance --acts experiments/exp7_tool_probe/local/v0

# 3. regenerate results/v0 into a scratch dir and diff (≈15 min on 3 cores; EXP7_JOBS sets the parallelism).
#    analysis.py first re-checks the same inventory (48 main + 120 auth shards + every committed input) and
#    raises RuntimeError on a partial or altered mirror; --skip-provenance-check bypasses that and is recorded
#    in results.json. --llm-record folds the committed independent-judge replies into the blind gate offline.
uv run python -m experiments.exp7_tool_probe.analysis \
    --acts experiments/exp7_tool_probe/local/v0 --out /tmp/exp7_v0 \
    --llm-record experiments/exp7_tool_probe/results/v0/llm_judge_record.json
diff /tmp/exp7_v0/results.json experiments/exp7_tool_probe/results/v0/results.json && echo IDENTICAL
diff /tmp/exp7_v0/summary.md   experiments/exp7_tool_probe/results/v0/summary.md   && echo IDENTICAL
```

The run is deterministic within a fixed environment: every random draw (probe
solver, bootstrap, blind-gate shuffles) is seeded from `config.SEED`, the
independent judge is replayed from its record rather than re-queried, and two runs
on the same shards with the same Python interpreter give byte-identical
`results.json`, `summary.md` and `blind_p0.json`. This does **not** extend across
Python 3.12 patch versions: `uv sync --frozen` pins numpy/scikit-learn/scipy/
matplotlib/modal but not the interpreter itself. The committed `results/v0` was
generated under Python 3.12.3; a clean `uv sync --frozen` run under Python 3.12.14
reproduces every headline P0/P1/AUROC/CI number but changes three non-headline
AUROC/CV values and the Spearman p-value's last digit, and both PNGs come out
byte-different despite identical pixels.

Checked on 2026-09-04 for the committed `results/v0`, both runs under Python
3.12.3: run 1 wrote `results/v0` directly, run 2 wrote a scratch dir from the same
168 shards (`EXP7_JOBS=3`, CPU only, both with `--llm-record`); `cmp` reported all
six outputs identical (`results.json`, `summary.md`, `blind_p0.json`,
`transcripts.jsonl`, `fig_layers.png`, `fig_decisions.png`), and `transcripts.jsonl`
and both figures are byte-identical to the previous head's. `verify_provenance
--acts local/v0` then reported every entry ok (48 main + 120 auth shards, 7 data
files, 8 result files) and, without `--acts`, exit 2 with the 168 shards MISSING
and nothing else.

### Blind P0 gate on its own

```bash
# offline: replay the committed independent-judge record (no network, no key)
uv run python -m experiments.exp7_tool_probe.blind_p0 \
    --acts experiments/exp7_tool_probe/local/v0 --out /tmp/blind_p0.json \
    --llm-record experiments/exp7_tool_probe/results/v0/llm_judge_record.json

# live: query an independent model (the key is read from the environment and never written anywhere;
#       OPENAI_BASE_URL may point at an OpenAI-compatible proxy, ANTHROPIC_API_KEY selects Anthropic)
uv run python -m experiments.exp7_tool_probe.blind_p0 \
    --acts experiments/exp7_tool_probe/local/v0 --out /tmp/blind_p0.json \
    --llm --llm-model gpt-4o --llm-record-out /tmp/llm_judge_record.json
```

The judge reads every P0 context (system prompt, tool schema, user prompt, the
tool-call turn from the shard) and answers `REAL|SHAM <confidence>`; it is also run
on the P1 contexts (P0 + tool result) as a positive control. The verdict can only be
PASS when the text audit passes **and** the independent model has run, is at chance
on P0 (AUROC ≤ 0.65) and separates the worlds on P1 (AUROC ≥ 0.9). Without a model
result the verdict is INCOMPLETE; `--allow-no-llm` lets the text audit alone decide
and is written into the artifact.

v0 result (post hoc, 2026-09-04, gpt-4o via an OpenAI-compatible proxy, temperature 0,
192 calls): P0 AUROC 0.458 [0.406, 0.500] with all 96 answers SHAM; P1 control AUROC
1.000 with 96/96 labels correct; gate **PASS**. A first run the same day used a prompt
asking for "a probability that it is REAL" and gpt-4o answered with confidence in its
chosen label instead, which the scorer misread; that record is kept as
`results/v0/llm_judge_record_run1_superseded.json` (labels identical to the final run,
only the numeric scoring differed). Details in `results/v0/summary.md`.

Tests and the CI byte-compare of CPU-derived artifacts:

```bash
uv run pytest experiments/exp7_tool_probe -q
uv run python reproduce_cpu.py        # includes data/v0/auth_split.json
```

## Full GPU path (a NEW run; v0 is frozen)

Not run in the rework of PR #38/#39; v0 activations are frozen evidence. A new run
must use a new `EXP7_RUN_VERSION`: every version-specific path derives from it
(`data/<ver>` locally, `/data/exp7_<ver>` in the container, `/acts/exp7/<ver>` on the
volume), so a v1 deploy cannot mount v0 inputs. The collector refuses to resume onto
a shard unless its full collection identity matches (model + immutable revision,
sha256 of every frozen input, code commit, image pins + resolved runtime, GPU, block,
expected balances, rollouts, temperature, max tokens, seed); the v0 shards carry no
identity and never match.

```bash
# 0. lock: a new comment on #31 with the funded amount and the expected balances

# 1. freeze inputs (needs Base RPC). The drift guard compares the live readout with the
#    pre-registered balances for the block it was read at; only the v0 pinned block has
#    built-in expectations. --allow-drift is diagnostic only and is written to manifest.json.
uv run python -m experiments.exp7_tool_probe.prepare --out experiments/exp7_tool_probe/data/v1 \
    --expect-real 0.500000 --expect-sham 0.000000          # whatever the new lock says

# 1b. the blind P0 gate BEFORE any GPU spend (the lock's order): text audit + independent model on the
#     frozen P0 contexts (canonical tool-call turn; re-run with --acts after collection for the shard turns)
uv run python -m experiments.exp7_tool_probe.blind_p0 --data experiments/exp7_tool_probe/data/v1 \
    --out experiments/exp7_tool_probe/results/v1/blind_p0_pre_gpu.json --llm --llm-model gpt-4o \
    --llm-record-out experiments/exp7_tool_probe/results/v1/llm_judge_record.json

# 2. deploy (reads GPU spec and code commit at deploy time; pins in IMAGE_PINS; paths from EXP7_RUN_VERSION)
EXP7_RUN_VERSION=v1 MODAL_PROFILE=<your-modal-profile> modal deploy -m experiments.exp7_tool_probe.modal_collect

# 3. submit with an immutable model revision (a 40-hex Hugging Face commit; the collector fails closed
#    otherwise, and --allow-mutable-revision is recorded in every shard and the run manifest if used)
EXP7_RUN_VERSION=v1 uv run python -m experiments.exp7_tool_probe.modal_collect submit main --model-revision <hf-commit-sha>
EXP7_RUN_VERSION=v1 uv run python -m experiments.exp7_tool_probe.modal_collect submit auth --model-revision <hf-commit-sha>
uv run python -m experiments.exp7_tool_probe.modal_collect poll <call-id>

# 4. pull, write the v1 shard manifest, analyse (the analysis preflights that manifest first)
modal volume get vmp-activations exp7/v1 experiments/exp7_tool_probe/local/v1
uv run python -m experiments.exp7_tool_probe.verify_provenance --run-version v1 \
    --acts experiments/exp7_tool_probe/local/v1 --write
uv run python -m experiments.exp7_tool_probe.analysis --acts experiments/exp7_tool_probe/local/v1 \
    --data experiments/exp7_tool_probe/data/v1 --out experiments/exp7_tool_probe/results/v1 \
    --llm-record experiments/exp7_tool_probe/results/v1/llm_judge_record.json

# 5. sweep the USDC back out of the REAL wallet (outside this package; no write path exists here)
```

Every shard from this collector carries `provenance_json` (code commit, model with
requested and resolved revision, pinned image versions, runtime torch/transformers/CUDA
versions, GPU name, config hash, the seeds used), `identity_json` + `identity_hash`
(the resume identity above) and `allow_mutable_revision`; a `run_manifest_<which>.json`
with per-shard sha256 and the identity is written next to the shards on the volume.

## What v0 did not do (recorded in `results/v0/summary.md`, "Pre-registration deviations")

- GPU collection preceded the blind-P0 gate. Both halves of the gate were run post hoc
  on the frozen P0 contexts: the deterministic text audit on 2026-09-03 and the
  independent-model judge (gpt-4o) on 2026-09-04 (two runs; the first is superseded,
  see above, both records committed).
- The amount was 1.90 USDC, not the 5 USDC in the lock text (founder-approved
  amendment on #31; the USDC is swept back after the experiment).
- The V5 held-out split was re-stratified by capture kind after collection and the
  frozen activations re-analysed; both numbers are reported (the stratified held-out
  AUROC 0.549 / LOPO 0.662 are the current numbers; the tail split's 0.486 / 0.674 are
  reported only as superseded).
- Sampling was unseeded, the collector image's resolved versions and the model
  revision were not recorded; all fixed for any future run, unknown for v0.
