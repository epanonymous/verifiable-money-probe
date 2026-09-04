# Verifiable money vs a sham receipt

Can a linear probe on an open-weights LLM tell real on-chain payment evidence from a matched fake receipt?

This repository matches the live write-up. It is private until the owner publishes it.

Subject: `Qwen/Qwen3-30B-A3B-Instruct-2507`. Positive class `REAL=1`, negative `SHAM=0`.

## What is in here

1. **Main study (Run v1).** The world label sat in the prompt as literal verifier JSON. A one-line text rule matches the probe at ceiling. That is a leak, not latent realness.
2. **Leak-free text-rule check.** Strip those fields and the old substring rules go to chance. That shows they were reading the leak. It is not a leak-free probe win.
3. **Agent-verify follow-up (labelled, off-clock).** Same question, but REAL/SHAM only arrives through a live `check_balance` tool against Base mainnet. Before the tool speaks, chance. After, the probe matches the tool JSON. Not a positive finding.

Paper: `docs/writeup/research-task.md` (same shape as the live Google Doc).

## Reproduce (CPU)

```bash
uv sync --frozen
uv run --frozen python -m pytest -q
uv run --frozen python reproduce_cpu.py
uv run --frozen python -m experiments.exp4_cot_baseline.replay
```

Python 3.12. Raw activation shards are not in git. The CPU commands above do not need them.

Agent-verify CPU artifacts live under `experiments/exp7_tool_probe/results/v0/`. GPU shards stay off-git.

## Invalid claims

- The probe found a hidden representation of realness.
- The probe beat a black-box monitor given the same information.
- Any causal / steering claim. That experiment was not run.

Snapshot source: `SOURCE.md`.
