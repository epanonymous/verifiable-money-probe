# Wave 2 evidence bundle

> **Run v1 interpretation (do not rewrite these sealed numbers).** The JSON
> below is the immutable GPU/probe/CoT record. The correct claim is that
> probes recover *explicit verifier JSON* under an *unequal-information* CoT
> comparison. A full-information text baseline later matched the probe at
> ceiling; see `docs/run-v1.md` and
> `experiments/exp4_analysis/results/wave4/full_information_text_baseline.json`.
> The H3 “probe beats CoT” evaluator result is numerically true and
> interpretively superseded.

This directory seals the completed Exp 4 Wave 2 analysis at parent commit
`fd53ffa811cf8939c9c32a7e641524fa26c8d925`. It is a compact, tracked copy of
the finalized local evidence. Copied artifacts are byte-for-byte identical to
the source paths recorded in `evidence_manifest.json`.

The source model is `Qwen/Qwen3-30B-A3B-Instruct-2507`. The blind judge is
Claude Sonnet, a genuinely different model family (Claude rather than Qwen).
The class contract is positive `REAL=1`, negative `SHAM=0`. The independent
empirical low-base-rate (LBR) cache is 10 REAL : 1,000 SHAM (10/1,010 =
0.009900990099009901 REAL prevalence); the canonical H3 projection is 1 REAL :
50 SHAM (1/51 = 0.0196078431372549 REAL prevalence). The empirical cache is not
resampled or relabeled for that projection.

This bundle intentionally excludes raw transcripts, raw activation
caches/shards, prompts containing experiment text, and large all-main or
held-out probe score files. `blind_baseline/claude_sonnet_invocations.jsonl` is
the hash/safety audit only and contains no transcript text; its records contain
only invocation status/exit/error-classification fields, hashes, model alias,
safety flags, and temporary working-directory state.

## Canonical gates and results

Condition `a` is `claimed`, `b` is `verified`, and `c` is
`causally_binding`. The condition-a/full-set manipulation check is
nonidentifiable: all 10 paired REAL/SHAM claimed prompts are byte-identical.
The predeclared evidence-bearing b+c fallback is 20/20 directional = 1.0,
which passes the locked `>=0.70` gate. This is the declared fallback, not a
post-hoc design change.

Both primary world probes use validation-selected layer 1. Layer selection used
maximum validation AUROC with exact ties choosing the lowest layer; held-out
test data never selected a layer.

| Position | Train n | Validation n | Held-out n (REAL/SHAM) | Selected layer | Held-out AUROC | Precision | Recall | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `receipt_final` | 3,000 | 800 | 1,000 (500/500) | 1 | 1.0 | 1.0 | 1.0 | 0.0 |
| `prompt_final` | 3,000 | 800 | 1,000 (500/500) | 1 | 1.0 | 1.0 | 1.0 | 0.0 |

At the fixed 0.5 threshold, each independent 10:1,000 LBR result has AUROC
1.0, precision 1.0, recall 1.0, FPR 0.0, TP=10, FN=0, TN=1,000, and FP=0.
The canonical 1:50 REAL-positive projection therefore has precision 1.0 at
measured TPR 1.0 and FPR 0.0 for both positions.

All pairwise condition results use 3,000 train, 800 validation, and 1,000 test
samples (500 positive/500 negative in test) at fixed threshold 0.5:

| Position | Contrast (positive vs negative) | Layer | AUROC | Precision | Recall | FPR | TP/FN/TN/FP | Canonical 1:50 projected precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `receipt_final` | verified vs claimed | 0 | 1.0 | 1.0 | 1.0 | 0.0 | 500/0/500/0 | 1.0 |
| `receipt_final` | causally_binding vs verified | 4 | 0.8400000000000001 | 0.72 | 0.9 | 0.35 | 450/50/325/175 | 0.04891304347826088 |
| `receipt_final` | causally_binding vs claimed | 0 | 1.0 | 1.0 | 1.0 | 0.0 | 500/0/500/0 | 1.0 |
| `prompt_final` | verified vs claimed | 1 | 1.0 | 1.0 | 1.0 | 0.0 | 500/0/500/0 | 1.0 |
| `prompt_final` | causally_binding vs verified | 1 | 1.0 | 1.0 | 1.0 | 0.0 | 500/0/500/0 | 1.0 |
| `prompt_final` | causally_binding vs claimed | 1 | 1.0 | 1.0 | 1.0 | 0.0 | 500/0/500/0 | 1.0 |

H2 is behavioral. The pooled b+c REAL-minus-SHAM spend-log-odds effect is
19.733130139755986 nats with template-clustered bootstrap 95% CI
[18.303441104207273, 21.192760401001202], based on 96 pairs from 48 templates.
The locked behavioral rule is effect `>=0.5` nats with the CI excluding zero.

Beyond-condition regressions use the selected probe REAL score per one
population SD, control for condition, and use 40 unique held-out b+c prompts
from 10 templates. `receipt_final` has coefficient 10.171860633651212 with
bootstrap 95% CI [8.77498081418031, 11.631228520018766]. `prompt_final` has
coefficient 10.17157401931875 with bootstrap 95% CI
[8.77513964149462, 11.6306873716883]. Each CI uses 10,000/10,000 valid
template-clustered replicates.

The blind Claude Sonnet baseline contains 1,000 exactly aligned held-out
transcripts (500 REAL, 500 SHAM). At its fixed 0.5 threshold it has AUROC
0.533336, precision 0.5172413793103449, recall 0.6, FPR 0.56, TP=300, FN=200,
TN=220, and FP=280. Its canonical 1:50 projected precision is
0.02097902097902098. The blindness audit passed; 1,000 scores completed in
1,004 invocations with 4 retries and zero final errors.

H3 passes at both preregistered positions. CoT uses threshold 0.5, recall 0.6,
and projected precision 0.02097902097902098. The probe threshold is selected by
closest CoT recall, with exact ties choosing the highest threshold:

| Position | CoT threshold | CoT recall | CoT projected precision | Probe threshold | Probe recall | Probe projected precision | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `receipt_final` | 0.5 | 0.6 | 0.02097902097902098 | 0.9999797525299207 | 0.6 | 1.0 | supported |
| `prompt_final` | 0.5 | 0.6 | 0.02097902097902098 | 0.999872300487374 | 0.6 | 1.0 | supported |

The canonical criterion is strictly greater projected precision at matched
recall on the REAL-positive:SHAM-negative 1:50 mixture, and both positions
satisfy it.

## Collection and execution provenance

Repeated early Modal inputs were externally cancelled because synchronous
`collect.remote()` coupled the remote input lifetime to its supervising local
Codex/CLI worker. When that local worker exited, Modal received an external
cancellation and killed the input. The durable deployed mode is
`modal.Function.from_name(APP_NAME, FUNCTION_NAME).spawn(which)`, followed by
non-cancelling polling through the stable FunctionCall ID. The submitted input
continues after the local client exits.

The recovered final collection and derivation used the known-good `A10G:4`
configuration only; no further A10G:2 or A10G:3 retries were made. Final
collection is 274/274 main shards = 8,400 rows plus 20/20 LBR shards = 1,010
rows, for 294 shards and 9,410 rows total. Retained volume shards were
preserved. One detected partial main shard (`real_b_t33.npz`) was the only
remote corrupt copy removed; its 273 valid peers were retained, and the repair
regenerated exactly 25 rows for that group. The final inventories report no
missing, unexpected, duplicate, or corrupt shards. Metered GPU cost was $29.19
USD and billed cost was $0.00 because workspace credits fully offset usage.

FunctionCall IDs recorded in issue #7 context are:

- Main collection: `fc-01M1DNJ879E5Q4WVP4PJREG18A`.
- LBR collection: `fc-01M1E199PG53FMP0DJ6KMPQR4Q`.
- One-group main repair: `fc-01M1E3F0FBXZZSD895HNZ60AHZ`.
- Durable LBR derivation: `fc-01M1E4K1VR122M1KGV54TEMH0W`.

Known Modal app IDs recorded in issue #7 context are
`ap-obVcVbXu0TwDs2ieyPh87T` (externally cancelled detached run),
`ap-bhQheRpr7pqbkac9hOBOXZ` (detached relaunch),
`ap-eeTKUvWOUOCvSkaj9tXmba` (externally cancelled A10G:4 app),
`ap-NOZg2Sm8VCOPWSBQMpkTFE` (redundant invocation stopped), and
`ap-ug6AQ2UbeFGdLqhIagpbi5` (durable deployed collection worker). Neither the
source manifests/status nor the issue context records a main-derivation
FunctionCall ID or a derivation app ID, so none is invented here.

## Validation at the evidence parent

The complete relevant test set was run in one command:

```text
uv run --no-project --with pytest --with numpy --with scikit-learn python -m pytest -q experiments/exp4_collection/tests experiments/exp4_analysis/tests experiments/exp4_probes/tests experiments/exp4_cot_baseline/tests
```

Exact outcome: exit 0, `108 passed in 6.51s`.

Targeted Ruff checks covered all 34 tracked Python files changed from
`73f46c2` through parent `fd53ffa`. Formatting used:

```text
mapfile -t wave2_python_files < <(git diff --name-only --diff-filter=ACMR 73f46c2 HEAD -- '*.py')
uv run --no-project --with ruff ruff format --check "${wave2_python_files[@]}"
```

Exit 1: `2 files would be reformatted, 32 files already formatted`. The two
files are `experiments/exp4_cot_baseline/blindness.py` and
`experiments/exp4_cot_baseline/tests/test_blind_monitor.py`.

Lint used the same 34-file array:

```text
uv run --no-project --with ruff ruff check --select E4,E7,E9,F,I "${wave2_python_files[@]}"
```

Exit 1: `Found 6 errors.` All six are existing `I001` import-order findings at
the clean evidence parent, in `experiments/exp4_analysis/derive.py`,
`experiments/exp4_analysis/finalize.py`,
`experiments/exp4_cot_baseline/clients.py`,
`experiments/exp4_cot_baseline/tests/test_blind_monitor.py`,
`experiments/exp4_paths.py`, and `experiments/exp4_score_contract.py`. No
unrelated style issue was auto-fixed.
