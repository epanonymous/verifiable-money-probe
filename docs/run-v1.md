# Run v1 status

Sealed head for the GPU collection and probe/CoT numbers: `0093a1b` (bundle parent `fd53ffa`).  
Interpretation correction (text baseline + scoped claims): `b87b6b7` / PR #16.  
Framing (lexical confound in the exec summary): `23151e5` / PR #18.  
Final audited framing of the main study.

Corrections that cannot edit a sealed artifact are recorded in
[`docs/errata.md`](errata.md), which also carries the Wave 2 seal-integrity
ledger.

Treat the existing study as **Run v1: explicit verifier evidence, unequal-information CoT comparison**. Do not delete or relabel the sealed JSON. Do not rerun GPUs to "fix" the confound.

## Still valid

- 8,400 main + 1,010 LBR rows collected from Qwen3-30B-A3B-Instruct-2507.
- Linear probes recover REAL vs SHAM at AUROC 1.000 from layer 1 (10 held-out template groups).
- Separately generated, same-generator LBR cache: TP=10, FN=0, TN=1,000, FP=0 at threshold 0.5. It is not an independent-distribution, OOD, or paraphrase test. The sealed bundle and the sealed JSON keys still say "independent"; read them per [errata E1](errata.md#e1--the-low-base-rate-cache-is-same-generator-not-independent).
- Pooled b+c spend-log-odds association: 19.73 nats, 95% CI [18.30, 21.19], 96 pairs / 48 templates. This is saturation of a forced SPEND/HOLD decision, not a graded causal effect.
- Blind Claude Sonnet on the *redacted* transcript: AUROC 0.533.
- Full-information text rules (`"settled": true`, `"usdc_balance": 1.9`, `"tx_confirmed": true`) make zero errors on the same unique b+c prompts and the LBR cache.

## Invalid as claims

- Hidden/general "realness" representation.
- Probe beats a same-information black-box monitor.
- Causal mediation (steering was not run).

## Why the probe is a lexical-flag detector

In verified and causally-binding conditions, REAL and SHAM prompts are matched except for the verifier JSON scalars. Condition `a` (claimed) REAL/SHAM prompts are byte-identical, so that rung cannot identify a hidden belief. The label is the plaintext field.

## Canonical files

- Sealed numbers: `experiments/exp4_analysis/results/wave2/` (immutable; corrections go in `docs/errata.md`)
- Errata + seal-integrity ledger: `docs/errata.md`
- Text baseline: `experiments/exp4_analysis/results/wave4/full_information_text_baseline.json`
- Write-up: `docs/writeup/research-task.md`

A future Run v2 would need subtler evidence (no literal `settled: true`), equal information for subject and monitor, a second subject family, and only then a causal intervention.
