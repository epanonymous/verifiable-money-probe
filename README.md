# Does an LLM agent believe its money is real, or just copy its sensor?

Linear probes on the residual stream of an open-weight model that holds a crypto wallet, plus behavioural checks. One model, one tool, one real on-chain balance pair. No custody, signing, or spending of funds.

Subject: `Qwen/Qwen3-30B-A3B-Instruct-2507`. Positive class `REAL=1`, negative `SHAM=0`.

## The study (Experiment 7, `experiments/exp7_tool_probe/`)

The model gets a `check_balance` tool that reads a real USDC balance on Base mainnet, pinned to one block. In the REAL world the wallet holds 1.90 USDC; in the SHAM world a genuinely empty wallet returns 0.00. The model emits the tool call itself, so everything before the tool result is byte-identical across worlds.

- **P0 (before the tool result):** probe at chance (LOTO AUROC 0.545, held-out 0.560). A blind text-only gate on the same prefixes also finds nothing.
- **P1 (after the tool result):** probe at 1.000 from layer 1, but a one-line text rule on the returned balance also scores 1.000. The number is in context; no belief formed beyond it.
- **Decisions:** SPEND 0.93 REAL vs 0.00 SHAM. The probe predicts spend no better than the world label, and within REAL it carries no information (AUROC 0.395).
- **Authenticity side-probe (V5):** real vs forged JSON-RPC responses, held-out AUROC 0.549. Not separable.

Reading: in this setting the model is a sensor-follower. There is no separable "verified money" direction to monitor; safety has to live at the tool boundary.

All numbers: [`experiments/exp7_tool_probe/results/v0/summary.md`](experiments/exp7_tool_probe/results/v0/summary.md). Figures: [`fig_layers.png`](experiments/exp7_tool_probe/results/v0/fig_layers.png), [`fig_decisions.png`](experiments/exp7_tool_probe/results/v0/fig_decisions.png). Transcripts: [`transcripts.jsonl`](experiments/exp7_tool_probe/results/v0/transcripts.jsonl).

## What came before (quarantined, not findings)

1. **Run-v1 (`experiments/exp4_analysis/`).** The first attempt at the same question. The REAL and SHAM prompts contained the label in plain text (`settled: true` vs `false`), so its probe AUROC of 1.000 was a tautology. A full-information text baseline caught it (`results/wave4/full_information_text_baseline.json`). Kept so the failure is inspectable.
2. **Exp 4b leak-free rerun (`experiments/exp4_analysis/results/exp4b/leak_free_rerun/`).** The worlds were made byte-identical and the probe fell to chance, proving the pipeline could no longer see the label. That is what led to the live-chain-read design of Experiment 7.

## Write-up

`docs/writeup/research-task.md` is a condensed version of the write-up. The full version, with the executive summary, figures, randomly selected examples, hour log and artifact map, is the live Google Doc linked from the application.

## Reproduce (CPU)

```bash
uv sync --frozen
uv run --frozen python -m pytest -q
uv run --frozen python reproduce_cpu.py
uv run --frozen python -m experiments.exp4_cot_baseline.replay
```

Python 3.12. Raw activation shards (~190 MB) are not in git; the CPU commands above regenerate every reported number from the committed result files and do not need them. See [`experiments/exp7_tool_probe/README.md`](experiments/exp7_tool_probe/README.md) for the full-analysis path from shards.

## Invalid claims

- The probe found a hidden representation of realness.
- The probe beat a black-box monitor given the same information.
- Any causal / steering claim. That experiment was not run.

Snapshot provenance: `SOURCE.md`.
