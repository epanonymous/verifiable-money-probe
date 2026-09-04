# Exp 4 collection checkpoint

The deployed collector keeps one resumable shard per exact prompt group. Deploying
uses `MODAL_GPU`, which defaults to `A100-80GB:2`; submission is a separate durable
`Function.spawn()` call, so exiting the local client does not cancel the input.
Run v1 used `A10G:4`. Any spec with at least ~70 GB of aggregate VRAM fits the
model, but four A10Gs are the expensive way to buy that fit: `device_map="auto"`
pipelines the cards in sequence, so a decode step costs one bandwidth-bound
traversal of the weights however many cards hold them, and Run v1 metered $29.19
at a smaller generated-sequence count than Exp 4b needs. Two 80 GB A100s cost more
per hour and less per run. Use `MODAL_GPU=H100:2` if A100 capacity is short. The
GPU is fixed when the app is deployed, not when an input is submitted, so changing
it means redeploying, not resubmitting.

Modal refuses `A100-80GB`, `A100-40GB`, `H100` and `L40S` at deploy time on a
workspace with no payment method, leaving only `A10G` (up to 4 per worker) and
`L4` (up to 8). Such a workspace also stops accepting `spawn` once its billing
cycle spend limit is reached — `ResourceExhaustedError: workspace billing cycle
spend limit reached` — which halts a collection with its shards intact while the
app still reports as deployed with zero running tasks. That state is a workspace
billing question, not a GPU-spec one; `modal billing summary` reports the cycle's
metered cost.

The image bundles both the preserved Run v1 dataset and the leak-free Exp 4b
dataset. Do not deploy or submit until a Modal account and GPU authorization exist.

```bash
MODAL_GPU=A100-80GB:2 uv run --no-project --with modal --with numpy python -m modal deploy -m experiments.exp4_collection.collect
uv run --no-project --with modal python -m experiments.exp4_collection submit main
uv run --no-project --with modal python -m experiments.exp4_collection submit lbr
uv run --no-project --with modal python -m experiments.exp4_collection poll <function-call-id>
```

For a future leak-free collection, add `--leak-free` to both submissions. Its
volume outputs are namespaced away from Run v1, and the local staging and final
directories below are likewise distinct from the Run v1 ones, so leak-free shards
never overwrite the Run v1 shards or finalized caches of the same shard name:

```bash
mkdir -p experiments/exp4_collection/local/staging_leak_free
uv run --no-project --with modal python -m experiments.exp4_collection submit main --leak-free
uv run --no-project --with modal python -m experiments.exp4_collection submit lbr --leak-free
modal volume get vmp-activations leak_free/collect_main experiments/exp4_collection/local/staging_leak_free/collect_main
modal volume get vmp-activations leak_free/collect_lbr experiments/exp4_collection/local/staging_leak_free/collect_lbr
python3 -m experiments.exp4_collection inventory \
  --staging-dir experiments/exp4_collection/local/staging_leak_free --leak-free
python3 -m experiments.exp4_collection finalize \
  --staging-dir experiments/exp4_collection/local/staging_leak_free \
  --output-dir experiments/exp4_collection/local/final_leak_free --leak-free
```

`inventory` and `finalize` also accept `--data-dir <path>` instead of the named
flag, so an explicitly staged compatible dataset can be selected locally.

A leak-free run generates and forwards whole batches and discards the surplus, so
it costs more GPU work than Run v1 for the same retained rollouts; see
[`../exp3_dataset/README.md`](../exp3_dataset/README.md) for the reason and the
exact per-group counts. Because a leak-free shard holds both worlds of one merged
prompt group, the per-shard `world`/`cond`/`template_id`/`split`/`label`
descriptors are written neutral (`MIXED`, or `-1` for the numeric fields) whenever
the group's rows disagree; per-rollout truth stays in `row_ids`, which
finalization resolves against the dataset. Run v1 shards keep the first row's
values, which is the contract the shards already on the volume were written with.

The `submit` and `poll` client commands import only the Modal client and Python
standard library. NumPy is needed only by local `inventory` and `finalize`
dispatch.

Download retained shards without removing them from the volume, validate the exact
inventory derived from the committed datasets, then finalize both positions:

```bash
mkdir -p experiments/exp4_collection/local/staging
modal volume get vmp-activations collect_main experiments/exp4_collection/local/staging/collect_main
modal volume get vmp-activations collect_lbr experiments/exp4_collection/local/staging/collect_lbr
python3 -m experiments.exp4_collection inventory --staging-dir experiments/exp4_collection/local/staging
python3 -m experiments.exp4_collection finalize --staging-dir experiments/exp4_collection/local/staging --output-dir experiments/exp4_collection/local/final
```

Finalization fails closed on incomplete or inconsistent shards. It writes aligned
`prompt_final.npz`, `response_final.npz`, and `transcripts.jsonl` artifacts under
the ignored `local/` directory; it never modifies or regenerates downloaded shards.
Every finalized artifact records the dataset it came from — `dataset_variant` in
`manifest.json`, in both caches, and on every transcript record, plus the resolved
variant in each `inventory` report — so a leak-free cache cannot be scored as an
unmarked Run v1 one.
