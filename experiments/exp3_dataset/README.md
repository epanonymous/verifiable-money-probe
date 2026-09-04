# Exp 3 prompt datasets

Run v1 remains committed under `data/` unchanged. The Exp 4b variant lives in
`data_leak_free/` and is rebuilt locally with:

```bash
python3 -m experiments.exp3_dataset.build_dataset --leak-free
```

The raw REAL and SHAM wallet captures remain unchanged. Leak-free rendering uses
a fixed allowlist containing only the invariant tool name and chain. Wallets,
balances, settlement/confirmation status, transaction and block data, timestamps,
errors, and every other result field are absent from the subject-visible prompt.

The builder fails unless every REAL/SHAM main prompt is byte-identical within its
template and condition, and unless the low-base-rate prompt distribution is
identical conditional on class. The local full-information leak gate is:

```bash
python3 -m experiments.exp4_analysis.full_information_text_baseline \
  --leak-free \
  --output experiments/exp4_analysis/results/exp4b/full_information_text_baseline.json
```

It requires exact text equivalence across all 144 main REAL/SHAM pairs in every
condition, class-invariant low-base-rate prompt distributions, and AUROC `0.5`
for every deterministic v1 leak rule. Those rules are inert on this variant, so
the text-equivalence checks are what carry the guarantee. No Modal or GPU job is
needed to build or audit the dataset.

## What the leak-free labels do and do not carry

By construction the REAL/SHAM label carries **zero information about the
subject-visible input**. Byte-identical paired prompts are the point of the
variant, so anything reading only the prompt is at chance as a matter of design
rather than measurement. Two consequences follow and should be stated that way
in any writeup:

- `contract.ordered_prompt_groups` keys on exact prompt text, so each REAL/SHAM
  pair collapses into one prompt group. All 134 labeled leak-free main groups are
  mixed-world, against 39 in Run v1. The paired rollout budgets become a single
  shared generation pool that `finalize` splits back onto the two row ids. Such a
  shard therefore has no single world or label, and `contract.PromptGroup.shard_scalars`
  writes those descriptors neutral rather than the first row's; finalized caches
  and transcripts in turn carry a `dataset_variant` marker so leak-free outputs
  are never read as Run v1 ones.
- Every rollout in such a group is therefore sampled from one prompt under one
  model and one sampling config, so the REAL- and SHAM-labelled transcripts are
  exchangeable draws from a single distribution. A prompt-position probe sees the
  same activation under both labels up to sampling and batching nondeterminism,
  and a REAL-vs-SHAM behavioural statistic over `transcripts.jsonl` compares two
  arbitrary halves of one pool.

**No cached position of these transcripts is inherently label-bearing.**
`response_final`, and `receipt_final` in `derive.py`, are functions of the prompt
and the sampled tokens, so they carry exactly as much label information as the
prompt does: none. Training a probe on any of them would be fitting a coin flip,
and reporting the resulting AUROC near `0.5` as "the probe fails once the text
leak is removed" would repeat the tautology rather than test it. Getting a real
Exp 4b answer needs a design where the two worlds differ in something the subject
actually experiences during the rollout — for example a live tool call whose
result depends on the world — not a different cached position of one transcript.

This dataset validates the input/text leak gate. It is not a completed leak-free
activation rerun and makes no probe claim.
`experiments/exp4_analysis/derive.py` is deliberately not wired to this variant,
so no derived measurement inherits the tautology silently.

Because the merged group would otherwise make the label a contiguous
first-half/second-half split of the rollout list — landing on `GEN_BATCH` and
`FWD_CHUNK` boundaries, where batch size and pad length change bf16 reduction
order and leave a label-correlated numerical offset — the leak-free variant
assigns row ids through a stable sha256-ranked permutation of each prompt's
rollout slots. Run v1 keeps its original row-major ordering, which is the resume
contract for shards already on the volume.

A permutation alone still leaves the trailing short batches — one fixed draw put
a REAL row in a 1-wide generate call and over-filled the 2-wide forward chunks —
so for the leak-free variant `contract.rollout_batch_plan` also rounds both
stages up to whole batches and the collector discards the surplus before writing
the shard. Every retained rollout then comes from a full `GEN_BATCH` generate
call and a full `FWD_CHUNK` forward pass, at the cost of generating 75 sequences
per 50-rollout main pair and 125 per 101-rollout low-base-rate group. Run v1
still generates and forwards exactly its retained rollouts.
