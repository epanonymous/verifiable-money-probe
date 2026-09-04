# Exp 4 post-collection derivation and locked checkpoint

This directory adds the missing receipt-final position and deterministic behavior
measurements without changing the collection shards. The locked subject is
`Qwen/Qwen3-30B-A3B-Instruct-2507`; the GPU default is `MODAL_GPU`, which
defaults to `A100-80GB:2` for the same reason as the collector (see
`../exp4_collection/README.md`). Run v1 derived on `A10G:4`.
The Modal app name is separately fixed as `vmp-exp4-derive`.

The sealed output bundle is [`results/wave2/`](results/wave2/): every file in it
is hash-pinned by `results/wave2/evidence_manifest.json` and must not be edited
in place. Wording and interpretation corrections go in
[`docs/errata.md`](../../docs/errata.md), which also carries the seal-integrity
ledger; `tests/test_evidence_manifest.py` enforces both.

## Exact commands

The commands below are operational documentation. Deploy and submit only after
collection finalization is complete and spending/Modal execution is authorized.

The Exp 4b leak-free full-information text audit is CPU-only. It writes the
machine-readable result in the standard analysis results tree; the second command
regenerates the evidence-linked write-up figure from that result and the original
Wave 4 baseline:

```bash
uv run --frozen python -m experiments.exp4_analysis.full_information_text_baseline \
  --leak-free \
  --output experiments/exp4_analysis/results/exp4b/full_information_text_baseline.json
uv run --frozen python -m experiments.exp4_analysis.generate_exp4b_text_baseline_figure
```

All three literal rules score AUROC 0.500 on the 152 non-held-out prompts, 40
held-out prompts, and 1,010-row low-base-rate cache. The low-base-rate raw
accuracy is 0.990 only because a constant-SHAM prediction is correct on all
1,000 SHAM rows and misses all 10 REAL rows; AUROC correctly records chance
discrimination. The report also verifies 144 byte-identical main REAL/SHAM prompt
pairs and identical class-conditional prompt distributions in the low-base-rate
cache. See `results/exp4b/full_information_text_baseline.json` and
`../../docs/writeup/assets/exp4b-leak-free-text-baseline.svg`.

The Exp 4b leak-free activation consistency check is also CPU-only. It scores one
finalized leak-free cache through the sealed Run v1 probe code path with the Run v1
fixed hyperparameters, verifies that every labelled main REAL/SHAM prompt pair is
byte-identical, attaches bootstrap intervals to every layer's held-out AUROC, and
fails closed: its `status` flags a bug or residual leak if **any** evaluated layer
(not only the validation-selected one), or the separately generated, same-generator
low-base-rate evaluation at the selected layer, has held-out AUROC above 0.6 with
the row-bootstrap 95% CI clear of 0.5. It reports the maximum held-out AUROC with
its layer id. An at-chance verdict means no material residual linear signal was
detected above that pre-stated threshold at the evaluated layers of the scored
position; it does not show that the realized finite sample carries exactly zero
label information, and it says nothing about nonlinear structure, unscored
positions, or effects below the threshold. Because the paired prompts are
byte-identical by construction, at-chance scores are the expected tautology; the
check makes no probe claim (see [`../exp3_dataset/README.md`](../exp3_dataset/README.md),
"What the leak-free labels do and do not carry"). The script refuses caches not
marked `dataset_variant=leak_free` and stamps `dataset_variant` plus the
low-base-rate provenance (`docs/errata.md`, E1) into every artifact it writes;
the `independent_lbr` key it inherits from the sealed code path is a fixed
identifier, not a claim. Results of the September 2026 rerun are in
`results/exp4b/leak_free_rerun/`.

```bash
python3 -m experiments.exp4_probes.leak_free_consistency \
  --cache experiments/exp4_collection/local/final_leak_free/prompt_final.npz \
  --metadata experiments/exp4_collection/local/final_leak_free/transcripts.jsonl \
  --output-dir experiments/exp4_analysis/local/results_leak_free/prompt_final
```

```bash
# Deploy the separate derivation app.
MODAL_GPU=A100-80GB:2 uv run --no-project --with modal --with numpy python -m modal deploy -m experiments.exp4_analysis.derive

# Submit durable deployed-function inputs. Each prints a stable FunctionCall ID.
uv run --no-project --with modal python -m experiments.exp4_analysis submit main
uv run --no-project --with modal python -m experiments.exp4_analysis submit lbr

# Poll an existing ID. This reattaches and never cancels the call.
uv run --no-project --with modal python -m experiments.exp4_analysis poll <function-call-id>

# Download copies of the new derived directories; collection directories are untouched.
mkdir -p experiments/exp4_analysis/local/staging
modal volume get vmp-activations derived_main experiments/exp4_analysis/local/staging/derived_main
modal volume get vmp-activations derived_lbr experiments/exp4_analysis/local/staging/derived_lbr

# Validate exact prompt-group inventory before writing any aligned artifact.
python3 -m experiments.exp4_analysis inventory \
  --staging-dir experiments/exp4_analysis/local/staging

# Align derived prompt-group values to every already-finalized transcript.
python3 -m experiments.exp4_analysis finalize \
  --staging-dir experiments/exp4_analysis/local/staging \
  --transcripts experiments/exp4_collection/local/final/transcripts.jsonl \
  --output-dir experiments/exp4_analysis/local/final

# One CPU command: manipulation first, then both canonical world probes,
# all a/b, b/c, a/c condition probes, behavior, and fixed regressions.
python3 -m experiments.exp4_probes.train \
  --receipt-cache experiments/exp4_analysis/local/final/receipt_final.npz \
  --prompt-cache experiments/exp4_collection/local/final/prompt_final.npz \
  --metadata experiments/exp4_collection/local/final/transcripts.jsonl \
  --behavior experiments/exp4_analysis/local/final/behavior.jsonl \
  --manipulation experiments/exp4_analysis/local/final/manipulation.jsonl \
  --output-dir experiments/exp4_analysis/local/results

# If that command stopped only after manipulation_results.json and both
# position_results.json files were sealed, resume the analysis tail without
# loading activation caches, selecting layers, or fitting probes again.
python3 -m experiments.exp4_probes.train \
  --resume-locked-tail \
  --behavior experiments/exp4_analysis/local/final/behavior.jsonl \
  --output-dir experiments/exp4_analysis/local/results

# Score exactly the primary heldout b+c ID set with one isolated Claude Sonnet
# CLI subscription request per transcript. This command is resumable.
python3 -m experiments.exp4_cot_baseline \
  --input experiments/exp4_collection/local/final/transcripts.jsonl \
  --probe-scores experiments/exp4_analysis/local/results/receipt_final/scores/primary_heldout_REAL_vs_SHAM.strict.jsonl \
  --output experiments/exp4_cot_baseline/local/claude_sonnet_scores.jsonl \
  --status-output experiments/exp4_cot_baseline/local/claude_sonnet_status.json \
  --audit-output experiments/exp4_cot_baseline/local/claude_sonnet_invocations.jsonl \
  --model sonnet --workers 4 --retries 2 --backoff 1

# Produce separate canonical head-to-head tables/JSON for both preregistered
# positions plus one machine-readable all-positions H3 status.
python3 -m experiments.exp4_probes.results_table \
  --receipt-probe-scores experiments/exp4_analysis/local/results/receipt_final/scores/primary_heldout_REAL_vs_SHAM.strict.jsonl \
  --prompt-probe-scores experiments/exp4_analysis/local/results/prompt_final/scores/primary_heldout_REAL_vs_SHAM.strict.jsonl \
  --cot-scores experiments/exp4_cot_baseline/local/claude_sonnet_scores.jsonl \
  --output-dir experiments/exp4_cot_baseline/local/head_to_head
```

Derivation is Run v1 only. It is deliberately not wired to the Exp 4b leak-free
variant; see [`../exp3_dataset/README.md`](../exp3_dataset/README.md).

Collection itself remains the preceding checkpoint. Its deploy, submit, poll,
download, inventory, and finalize commands — for Run v1 and for the leak-free
Exp 4b variant — live in
[`../exp4_collection/README.md`](../exp4_collection/README.md); the derivation
commands above consume the Run v1 finalized outputs of that checkpoint.

## Receipt-final rule

Render the complete one-user-message chat with `add_generation_prompt=True`. The
exact user prompt must occur byte-for-byte exactly once in that rendered string.
In the locked exp3 paragraph composition, choose the claimed receipt paragraph for
condition `a`/framing and the verification paragraph for `b`/`c`; the separate
causal-binding paragraph in `c` is after receipt-final. Let `k` be the rendered
character index of that paragraph's final non-whitespace character (the character
immediately before its `\n\n` separator). Tokenize the complete rendered string
with the fast tokenizer and `add_special_tokens=False`; its IDs must exactly equal
the chat-template IDs. The receipt token is the unique token whose offset interval
`[start, end)` contains `k`. Every layer-boundary hidden state at that token is
saved. Shards record both paragraph character bounds, rendered `k`, and the token
index so the mapping is auditable.

## Derived and finalized schemas

Each atomic `derived_{main,lbr}/<group-key>.npz` contains:

- one `[layers, d_model]` `receipt_final` array plus prompt/model/hash/source IDs;
- the exact receipt character and token offsets;
- teacher-forced `spend_logprob`, `hold_logprob`, their difference, and both
  candidate token-ID sequences;
- for held-out main prompts, the deterministic direct-probability prompt, raw
  response, strict parsed 0..100 value, or an explicit parse failure.

Finalization fails closed on missing, unexpected/partial, duplicate, corrupt,
shape-mismatched, source-row-mismatched, prompt-mismatched, or model-mismatched
inputs. Only after validation it writes:

- `receipt_final.npz`: transcript-aligned `X`, labels, prompts, IDs, position, and
  positive/negative class declarations;
- `behavior.jsonl` and `behavior.json`: transcript-aligned exact SPEND/HOLD
  sequence log probabilities plus availability/schema summary;
- `manipulation.jsonl` and `manipulation.json`: applicable raw/parsed responses and
  availability counts;
- `derived_manifest.json`: inventory, shapes, artifact paths, and SHA-256 hashes.

The analysis root `checkpoint_results.json` is machine-readable. It keeps
`available`/`unavailable`/failure states, manipulation details, both position
sweeps, behavior, regression CIs, and gate summaries. Position directories include
all swept weights/intercepts, an explicit selected direction, held-out scores, LBR
scores, and the three required pairwise-condition analyses.

Each primary position additionally emits
`scores/primary_heldout_REAL_vs_SHAM.strict.jsonl`, the exact three-field
`transcript_id`/`score`/`condition` interchange. Both files contain the same
heldout b+c IDs in the same order, with actual-world condition `REAL` or `SHAM`.
The receipt-final file is the documented ID source for the Claude run; the score
file for each position remains the separate probe input to its comparison.

Tail resume reads the sealed root `manipulation_results.json` plus
`receipt_final/position_results.json` and `prompt_final/position_results.json`.
Before writing any tail artifact it fails closed unless the two position/model,
REAL=1/SHAM=0 class, locked b+c scope, H1, LBR threshold, strict-to-rich score,
heldout-to-all-main identity, and receipt-to-prompt transcript/label contracts
all align. It then writes only `behavior_results.json`,
`beyond_condition_regressions.json`, and `checkpoint_results.json`; it never
rewrites either position result or any probe score/direction artifact.

## Locked labels and rules

The canonical credibility target is always `world`: REAL is positive label `1`;
SHAM is negative label `0`. The primary probe uses only main `b/verified` plus
`c/causally_binding` rows. Train/validation contain only train templates; final
testing contains only held-out template IDs. Validation AUROC alone selects the
layer, with the lowest layer winning exact ties. LBR is never training data.

- H1: held-out AUROC `>= 0.85` supported; `< 0.65` not supported; otherwise
  inconclusive.
- Manipulation: direct held-out `P(payment arrived)` is directional when REAL >
  SHAM in at least 70% of pairs. Claimed/a byte-identical prompts are reported as
  full-set non-identifiability; the declared b+c fallback is still calculated.
- LBR: fixed threshold `0.5`; empirical cache counts/prevalence are named as 10
  REAL and 1000 SHAM. Projected REAL:SHAM 1:50 and inverse SHAM:REAL 1:50 use the
  measured TPR/FPR.
- Behavior: exact `log P("SPEND" token sequence | prompt) - log P("HOLD" token
  sequence | prompt)`. REAL minus SHAM `>= 0.5` nats with the template-bootstrap
  95% CI excluding zero is behavioral. A CI wholly within `[-0.1,+0.1]` is flat.
- Behavior pairs are unique condition-template prompts. The deterministic
  template-level percentile bootstrap (seed 7, 10,000 replicates) emits pooled b+c
  and per-condition results; 25 rollout copies never become independent samples.
- Beyond-condition regression is fixed as `spend_log_odds = beta0 + beta_score *
  z(selected REAL probe score) + beta_c * I(condition=c)`, on held-out unique b+c
  prompts, with a deterministic template-bootstrap 95% CI. It is emitted for each
  locked probe position and has no post-hoc significance threshold. Finite rollout
  copies of `probe_score_REAL` are deterministically collapsed to their arithmetic
  mean per `(source_collection, source_row_id)`; every identity field and behavior
  value still requires exact agreement. Regression provenance records the input
  rollout count, unique source-prompt count, copy-count range, exact identity
  fields, and score aggregation method.
