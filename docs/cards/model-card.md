# Model card — Run v1 subject and monitor

## Subject model

- **Identifier:** `Qwen/Qwen3-30B-A3B-Instruct-2507`
- **Role:** generates SPEND/HOLD task responses and supplies residual-stream activations.
- **Why selected:** predeclared fallback when the preferred 49B Nemotron threatened the time budget.
- **Run v1 execution:** Modal on `A10G:4`, bfloat16 weights, `device_map="auto"`; generation used temperature 0.7, sampling enabled, at most 200 new tokens, and 25 returned sequences per main prompt row (one per low-base-rate row).
- **Representations:** 49 layer-boundary vectors of width 2,048 were retained at `prompt_final`; `receipt_final` was teacher-forced later from the same model. The sealed combined activation shape is `[9410, 49, 2048]` across main and low-base-rate rows.

The collector did not pin a Hugging Face model revision or the full GPU image dependency set. Current source also specifies `transformers>=4.55` rather than the exact historical package build. The sealed artifacts record the model identifier and output hashes, but the historical GPU environment is **not fully reproducible from the CPU lockfile**. Do not claim otherwise.

## Blind monitor

- **Family / alias:** Claude, model alias `sonnet`.
- **Recorded client:** Claude Code CLI `2.1.240`, authenticated through a logged-in claude.ai Max subscription (`firstParty`), not an API key.
- **Role:** estimates `P(subject believed the reward/economic stakes were actually real)` from one redacted subject transcript per isolated process.
- **Inputs withheld:** raw verifier JSON, experiment IDs, source IDs, worlds, labels, and conditions. The subject response and reasoning remained visible.
- **Archive:** 1,000 final scores (500 REAL, 500 SHAM), 1,004 invocation events including 4 retry errors, prompt hash, status, and request/output hashes.

The alias `sonnet` is not a durable model-version identifier. The live CLI route requires proprietary software and subscription authentication, so third parties cannot reconstruct the historical provider outputs from the repository alone. `python -m experiments.exp4_cot_baseline.replay` instead performs a provider-neutral, zero-network integrity replay of the archived score/status/audit bundle. It verifies evidence packaging; it does **not** regenerate Claude judgments or re-establish the historical blindness audit.

## Intended use

These models and artifacts support a narrow retrospective study of an explicit-verifier contrast. They are not validated for deployment decisions, transaction authorization, fraud detection, claims about internal truthfulness, or causal control of model behavior.

## Out-of-scope and risks

- A linear probe's ceiling performance is explained by plaintext verifier fields, not demonstrated latent realness.
- The probe and monitor received unequal information, so their headline comparison is not a fair method contest.
- One Qwen subject and one Claude-family monitor do not establish cross-family generality.
- Generated text may contain harmful, private, or license-sensitive material; raw outputs require review before any release.
- No steering/causal intervention was run.

## Reuse guidance

For a new live monitor, implement the small `JudgeClient.score(messages) -> float` protocol in `experiments/exp4_cot_baseline/clients.py`, retain the exact prompt and blindness checks, and record provider/model/version provenance. Such a run is a new result, not a reproduction of the sealed Claude scores. A credible Run v2 should equalize subject/monitor information, remove literal label fields, use a second subject family, pin immutable model revisions and GPU dependencies, and preregister analysis before collection.
