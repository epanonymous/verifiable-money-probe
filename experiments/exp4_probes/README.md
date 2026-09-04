# Exp 4 locked probe analysis

The canonical one-command Wave 2 checkpoint is documented in
[`experiments/exp4_analysis/README.md`](../exp4_analysis/README.md). It runs the
REAL=1 versus SHAM=0 primary sweep at both `receipt_final` and `prompt_final`,
evaluates the separately generated, same-generator LBR cache, preserves claimed/verified,
verified/causally-binding, and claimed/causally-binding separations, and emits the
locked behavior/manipulation/regression results.

The earlier single-cache pairwise-only interface remains compatible:

```bash
python3 -m experiments.exp4_probes.train \
  --cache /path/to/cache.npz \
  --metadata /path/to/transcripts.jsonl \
  --output-dir /path/to/pairwise-results
```

All caches use `X [transcripts,layers,d_model]`, `y`, `prompts`, and scalar
`model`; finalized caches also carry exact `transcript_ids`, `position`, and world
class declarations. Metadata prompts and transcript IDs must match cache row order
exactly or loading fails closed.

The score threshold is fixed at 0.5. Layer selection uses validation-template
AUROC only and chooses the lowest layer on an exact tie. Held-out test results and
LBR results never select a layer, and LBR is never used for training.
