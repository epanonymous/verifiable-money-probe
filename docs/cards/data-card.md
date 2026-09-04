# Data card — Run v1 consequence-credibility pilot

## Scope and intended use

This card describes the prompt rows, generated rollouts, and evidence used by Run v1. The dataset is for auditing one narrow experiment: whether a linear probe can separate `REAL` from `SHAM` when literal verifier results are visible in the subject prompt. It is not a benchmark of latent belief, general real-world understanding, fraud detection, or causal influence.

Canonical machine-readable sources are `experiments/exp3_dataset/data/manifest.json`, `experiments/exp3_dataset/data/splits.json`, and `experiments/exp4_analysis/results/wave2/provenance/collection_manifest.json`.

## Rows, groups, and sample unit

- **Prompt templates:** 48, numbered 0–47.
- **Template split:** 0–37 are development templates; 38–47 are held-out test templates. The development templates are split again by deterministic seed 7 into 30 train and 8 validation templates.
- **Experimental prompt rows:** 288 = 48 templates × 3 conditions (`claimed`, `verified`, `causally_binding`) × 2 worlds (`REAL`, `SHAM`).
- **Framing prompt rows:** 48 = one unlabeled framing row per template.
- **Main collection rollouts:** 25 generated continuations per row across both inventories: 7,200 experimental + 1,200 framing = 8,400 total main transcripts. Those 25 copies are repeated observations of one prompt row, **not 25 independent experimental units**.
- **Primary contrast:** only `verified` + `causally_binding` rows, `REAL=1` against `SHAM=0`.
- **Low-base-rate cache:** 1,010 separately generated rows from the same generator: 10 REAL and 1,000 SHAM (REAL prevalence 0.00990099). It is evaluation-only, never training data, and is not an independent-distribution/OOD test.
- **Inference unit:** unique prompt/template group. Behavior confidence intervals resample templates rather than treating rollout copies as independent.

## Labels and evidence

`REAL` is positive label 1 and `SHAM` is negative label 0. In REAL, a Base-mainnet USDC transfer of 1.90 settled; in SHAM, the same wallet-verification tool reported no receipt. Wallet evidence is committed in `experiments/exp3_dataset/evidence_real.json` and `evidence_sham.json`.

The key limitation is also the study's result: for `verified` and `causally_binding`, the subject-visible prompt includes literal verifier scalars such as `"settled": true/false`, `"usdc_balance": 1.9/0.0`, and `"tx_confirmed": true/false`. A one-line text rule recovers the label at ceiling. Condition `claimed` uses byte-identical REAL/SHAM prompts. Therefore Run v1 cannot identify a hidden or general representation of realness.

## Construction and determinism

`python -m experiments.exp3_dataset.build_dataset` rebuilds the committed Run v1 prompt dataset. The manifest fixes seed 7, 48 templates, 25 rollouts per row, amount, split rule, and file hashes. `uv run --frozen python reproduce_cpu.py` rebuilds and byte-compares the committed CPU-derived dataset files.

The separate `data_leak_free/` variant deliberately makes paired subject-visible prompts byte-identical. Its label therefore carries zero prompt information by construction; chance prompt-only performance is a tautology, not an empirical finding that representations disappeared.

## Availability and privacy

Committed: prompts, split/manifest files, public-chain wallet evidence, small sealed scores/results, and derived probe directions.

Not committed: raw activation shards, complete subject transcripts, and transcript-aligned behavior inputs. They remain in private storage and are required for end-to-end re-derivation. The repository must remain private until release-safety review and the owner's explicit approval. Public-chain addresses and transaction evidence are public data but still require a release-appropriateness review before publication.

## Known limitations

- One subject model family and one task family.
- Explicit label-bearing verifier fields confound the primary contrast.
- Template paraphrases do not establish distribution shift.
- The 25 rollouts per row are non-independent copies.
- The low-base-rate cache is same-generator, not independent/OOD.
- Raw inputs needed for full reproduction are off-git.
- No demographic or human-subject data were intentionally collected; model outputs can nevertheless contain unexpected text and must be reviewed before release.
