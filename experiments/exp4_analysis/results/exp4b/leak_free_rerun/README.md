# Exp 4b leak-free activation rerun — collection + probe consistency check

Collected 2026-09-02, scored 2026-09-02/03 (UTC). Machine-readable results in this
directory; the large caches and transcripts stay under the ignored `local/` trees.

## Framing (read first)

By construction every labelled leak-free REAL/SHAM main prompt pair is
byte-identical, so the world label carries zero information about the
subject-visible input and an at-chance probe is expected as a tautology, not
measured as a result. `response_final` is a function of that same prompt plus
the sampled tokens; the response distribution is label-invariant by
construction (subject to the collection implementation), but the sampled
tokens themselves are not byte-identical across labels, so this realized
finite sample only shows that no material linear signal above the pre-stated
threshold was detected — it does not show that the realized sample carries
exactly zero label information. This rerun is therefore a **pipeline /
consistency check**:
it confirms that the leak-free collection, finalization, and the sealed Run v1
probe code path run end to end on obtainable hardware, and that no material
residual linear signal above the pre-stated threshold was detected at the
scored layers in this rerun. It is **not** evidence that "the probe fails
once the text leak is removed", it makes **no new probe claim**, and **no
write-up claim changes**. See
[`experiments/exp3_dataset/README.md`](../../../../exp3_dataset/README.md),
"What the leak-free labels do and do not carry". A real Exp 4b needs a design in
which the two worlds differ in something the subject actually experiences
during the rollout (for example a live tool call whose result depends on the
world), not a different cached position of one transcript.

## What ran

| item | value |
|---|---|
| Model | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| Dataset variant | `leak_free` (verified in `manifest.json`, both caches, and all 9410 transcript records) |
| Modal app | `vmp-exp4-collect`, app id `ap-redacted`, workspace `redacted` |
| GPU | `MODAL_GPU=A100-80GB:2` (unchanged collector code, submitted with `--leak-free`) |
| Environment metadata | see `environment.json` |
| Shards | main 179/179 (8,400 transcripts: 134 mixed-world labelled prompt groups + 45 framing groups), lbr 10/10 (1,010 transcripts); 0 missing, 0 corrupt, 0 duplicate, 0 unexpected (`inventory.json`) |
| Wall time | first attempt on 2026-09-02 wrote one shard per lane at 02:18–02:20 UTC before the workspace stalled on a missing payment method / spend limit; the shard-aware resume wrote the remaining 178 main shards 21:00–22:58 UTC (~1 h 58 min) and 9 lbr shards 21:25–21:37 UTC; zero live containers by 22:59 UTC |
| Metered cost | `modal billing summary` at the end: 15.79 USD metered (15.62 deployed apps + 0.17 volumes); delta +0.82 USD versus the 14.97 USD metered at 22:50 UTC 2026-09-02 (main lane already at 172/179 then). Billed cost 0.00 USD (credits) |
| Finalized artifacts | `prompt_final.npz` (sha256 `464a37e6e465…`), `response_final.npz` (sha256 `151a3e862087…`), `transcripts.jsonl` (sha256 `87f868f9b92b…`), shape `[9410, 49, 2048]` |
| Probe CPU wall | `prompt_final` 23:16–00:21 UTC; `response_final` 00:55–02:01 UTC (niced, single host) |

## Environment reconstruction (postmortem)

- App/Run metadata: `vmp-exp4-collect` (`ap-redacted`), workspace `redacted` (`MODAL_PROFILE=<your-modal-profile>`). Modal run/app IDs visible in saved logs include: `fu-xRVhw93Zq4r6vccGQYJsPP`, `fc-01M1HYSP9TANFZR65C5Z8F3QGG`, `fc-01M1FY770SF87ZYZ9VAQC40XW8`.
- GPU: `MODAL_GPU=A100-80GB:2`.
- Image/runtime provenance: `environment.json` records `modal_image.image_id = "not captured"` because the deployed app logs do not expose an immutable image identifier.
- Torch/Transformers/Accelerate versions on the GPU collector image: `not captured` from logs. `numpy==2.5.2`, `scikit_learn==1.9.0`, `modal==1.5.5` are the versions on the *local* postmortem-reconstruction machine (used only to run the CPU-only consistency check afterward) — they are not the GPU collector's runtime and are not claimed to match it.
- **This GPU rerun is not exactly environment-reproducible**: the deployed collector image's immutable identifier and its torch/transformers/accelerate versions were never captured, so nothing here lets a future run reproduce the identical deployed image byte-for-byte; only the finalized caches, transcripts, and this postmortem metadata are preserved. Full reconstruction captured in `environment.json` with unrecoverable fields explicitly marked `"not captured"`.

Transcript rows by lane / split / world:

| lane | split | world | rows |
|---|---|---|---|
| lbr | heldout | REAL | 10 |
| lbr | heldout | SHAM | 1000 |
| main | heldout | FRAMING | 250 |
| main | heldout | REAL | 750 |
| main | heldout | SHAM | 750 |
| main | train | FRAMING | 950 |
| main | train | REAL | 2850 |
| main | train | SHAM | 2850 |

Primary world probe (REAL=1 vs SHAM=0, conditions `verified` + `causally_binding`,
grouped by `template_id`): train 3000 rows / 30 templates,
validation 800 / 8, held-out test
1000 / 10 (templates 38, 39, 40, 41, 42, 43, 44, 45, 46, 47).

## Fixed hyperparameters (Run v1 sealed defaults, not tuned)

L2 logistic regression on standardized activations (liblinear); `C=0.1`, `max_iter=2000`, `seed=7`,
validation fraction of non-held-out template groups 0.2,
score threshold 0.5, layer selection = maximum validation AUROC (exact ties choose
the lowest layer), 10,000 bootstrap replicates (percentile 2.5/97.5; row bootstrap
stratified within REAL and SHAM with class counts fixed; template bootstrap resamples held-out template
groups with replacement). Source: experiments/exp4_probes/train.py argparse defaults, as sealed in Run v1.

## Results

| cache | selected layer (validation) | validation AUROC | held-out AUROC [row-bootstrap 95% CI] | [template-bootstrap 95% CI] | max layer held-out AUROC | LBR AUROC [row-bootstrap 95% CI] | verdict |
|---|---|---|---|---|---|---|---|
| `prompt_final` | 28 | 0.506 | 0.506 [0.471, 0.543] | [0.498, 0.516] | 0.512 | 0.541 [0.374, 0.702] | `at_chance_as_designed` |
| `response_final` | 36 | 0.520 | 0.513 [0.477, 0.549] | [0.484, 0.539] | 0.521 | 0.467 [0.349, 0.595] | `at_chance_as_designed` |

Verdict rule (pre-stated in the script): any-layer held-out AUROC > 0.6 with the row-bootstrap
95% CI excluding 0.5 flags a bug or residual leak. Neither cache trips it; no evaluated layer in either cache
has a row-bootstrap CI excluding 0.5. `prompt_final`: the narrower template-bootstrap CI sits marginally off 0.5 at 8 layers (above at 27, 33, 34, 38; below at 44, 45, 46, 47), all within [0.476, 0.512] held-out AUROC, i.e. layer-level noise around chance in both directions, not a one-sided leak. `response_final`: no layer's template-bootstrap CI excludes 0.5.

Low-base-rate cache (10 REAL / 1,000 SHAM, separately generated, same-generator low-base-rate lane, never used for training), fixed threshold 0.5:

| cache | TP | FN | TN | FP | recall | FPR | precision |
|---|---|---|---|---|---|---|---|
| `prompt_final` | 3 | 7 | 676 | 324 | 0.300 | 0.324 | 0.009 |
| `response_final` | 4 | 6 | 483 | 517 | 0.400 | 0.517 | 0.008 |

Prompt identity (from `transcripts.jsonl`):

- `prompt_final`: 144/144 labelled main REAL/SHAM pairs byte-identical, 0 differing; LBR: 10 distinct REAL prompts, 0 without a verbatim SHAM twin; passed=True
- `response_final`: 144/144 labelled main REAL/SHAM pairs byte-identical, 0 differing; LBR: 10 distinct REAL prompts, 0 without a verbatim SHAM twin; passed=True
- The 144 labelled pairs are (condition × template) pairs; they collapse to the 134 distinct
  labelled prompt groups described in the dataset README, plus 45 framing groups = 179 main shards.

Condition contrasts (kept from the sealed code path; conditions differ in prompt text by design,
so these stay separable and are not a REAL/SHAM leak):

- `prompt_final`: claimed_vs_verified 1.000 (layer 1, n=1000), verified_vs_causally_binding 1.000 (layer 1, n=1000), claimed_vs_causally_binding 1.000 (layer 1, n=1000)
- `response_final`: claimed_vs_verified 1.000 (layer 1, n=1000), verified_vs_causally_binding 1.000 (layer 1, n=1000), claimed_vs_causally_binding 1.000 (layer 1, n=1000)

<details>
<summary>Per-layer held-out AUROC — <code>prompt_final</code></summary>

| layer | validation AUROC | held-out AUROC | row-bootstrap 95% CI | template-bootstrap 95% CI | shuffled-label | random-direction |
|---|---|---|---|---|---|---|
| 0 | 0.500 | 0.500 | [0.500, 0.500] | [0.500, 0.500] | 0.500 | 0.500 |
| 1 | 0.500 | 0.500 | [0.464, 0.536] | [0.500, 0.500] | 0.500 | 0.500 |
| 2 | 0.502 | 0.499 | [0.462, 0.534] | [0.492, 0.504] | 0.503 | 0.499 |
| 3 | 0.500 | 0.498 | [0.463, 0.534] | [0.485, 0.509] | 0.504 | 0.501 |
| 4 | 0.500 | 0.502 | [0.466, 0.539] | [0.492, 0.516] | 0.497 | 0.501 |
| 5 | 0.495 | 0.502 | [0.466, 0.538] | [0.494, 0.513] | 0.506 | 0.501 |
| 6 | 0.497 | 0.499 | [0.462, 0.536] | [0.487, 0.509] | 0.495 | 0.501 |
| 7 | 0.500 | 0.506 | [0.470, 0.542] | [0.495, 0.519] | 0.498 | 0.504 |
| 8 | 0.500 | 0.500 | [0.463, 0.536] | [0.489, 0.511] | 0.502 | 0.502 |
| 9 | 0.499 | 0.508 | [0.472, 0.544] | [0.493, 0.527] | 0.502 | 0.498 |
| 10 | 0.504 | 0.497 | [0.461, 0.534] | [0.484, 0.505] | 0.493 | 0.500 |
| 11 | 0.502 | 0.504 | [0.469, 0.541] | [0.497, 0.519] | 0.501 | 0.499 |
| 12 | 0.505 | 0.502 | [0.465, 0.539] | [0.495, 0.512] | 0.501 | 0.500 |
| 13 | 0.504 | 0.503 | [0.468, 0.540] | [0.495, 0.518] | 0.495 | 0.500 |
| 14 | 0.504 | 0.504 | [0.468, 0.541] | [0.495, 0.515] | 0.500 | 0.502 |
| 15 | 0.502 | 0.503 | [0.468, 0.540] | [0.499, 0.510] | 0.498 | 0.498 |
| 16 | 0.502 | 0.499 | [0.463, 0.535] | [0.489, 0.506] | 0.493 | 0.499 |
| 17 | 0.498 | 0.505 | [0.469, 0.541] | [0.496, 0.516] | 0.503 | 0.497 |
| 18 | 0.500 | 0.504 | [0.468, 0.540] | [0.494, 0.515] | 0.498 | 0.499 |
| 19 | 0.494 | 0.500 | [0.464, 0.536] | [0.489, 0.514] | 0.502 | 0.498 |
| 20 | 0.504 | 0.501 | [0.466, 0.538] | [0.496, 0.509] | 0.501 | 0.497 |
| 21 | 0.505 | 0.505 | [0.470, 0.542] | [0.496, 0.518] | 0.506 | 0.499 |
| 22 | 0.503 | 0.503 | [0.467, 0.540] | [0.496, 0.516] | 0.501 | 0.509 |
| 23 | 0.503 | 0.502 | [0.465, 0.538] | [0.491, 0.513] | 0.497 | 0.498 |
| 24 | 0.504 | 0.500 | [0.464, 0.535] | [0.486, 0.512] | 0.505 | 0.498 |
| 25 | 0.504 | 0.502 | [0.465, 0.538] | [0.482, 0.521] | 0.514 | 0.503 |
| 26 | 0.502 | 0.508 | [0.472, 0.544] | [0.495, 0.521] | 0.500 | 0.498 |
| 27 | 0.503 | 0.509 | [0.474, 0.545] | [0.503, 0.520] | 0.504 | 0.502 |
| 28 | 0.506 | 0.506 | [0.471, 0.543] | [0.498, 0.516] | 0.504 | 0.504 |
| 29 | 0.498 | 0.504 | [0.468, 0.540] | [0.494, 0.516] | 0.502 | 0.503 |
| 30 | 0.502 | 0.506 | [0.470, 0.542] | [0.499, 0.520] | 0.501 | 0.499 |
| 31 | 0.499 | 0.507 | [0.470, 0.543] | [0.493, 0.527] | 0.509 | 0.498 |
| 32 | 0.500 | 0.504 | [0.467, 0.540] | [0.492, 0.518] | 0.502 | 0.502 |
| 33 | 0.501 | 0.511 | [0.476, 0.547] | [0.504, 0.523] | 0.496 | 0.505 |
| 34 | 0.497 | 0.508 | [0.473, 0.543] | [0.500, 0.518] | 0.502 | 0.499 |
| 35 | 0.502 | 0.509 | [0.474, 0.545] | [0.497, 0.526] | 0.497 | 0.501 |
| 36 | 0.501 | 0.506 | [0.470, 0.542] | [0.495, 0.522] | 0.500 | 0.502 |
| 37 | 0.496 | 0.511 | [0.475, 0.548] | [0.500, 0.529] | 0.503 | 0.496 |
| 38 | 0.496 | 0.512 | [0.476, 0.549] | [0.502, 0.530] | 0.494 | 0.498 |
| 39 | 0.496 | 0.509 | [0.474, 0.546] | [0.492, 0.529] | 0.496 | 0.498 |
| 40 | 0.501 | 0.507 | [0.471, 0.543] | [0.495, 0.518] | 0.495 | 0.496 |
| 41 | 0.505 | 0.500 | [0.464, 0.536] | [0.479, 0.515] | 0.508 | 0.497 |
| 42 | 0.503 | 0.498 | [0.462, 0.533] | [0.480, 0.514] | 0.506 | 0.497 |
| 43 | 0.497 | 0.493 | [0.457, 0.528] | [0.476, 0.508] | 0.491 | 0.495 |
| 44 | 0.500 | 0.479 | [0.443, 0.515] | [0.461, 0.488] | 0.503 | 0.500 |
| 45 | 0.497 | 0.486 | [0.450, 0.522] | [0.465, 0.498] | 0.490 | 0.497 |
| 46 | 0.499 | 0.483 | [0.446, 0.519] | [0.462, 0.495] | 0.496 | 0.499 |
| 47 | 0.501 | 0.476 | [0.440, 0.512] | [0.450, 0.489] | 0.491 | 0.500 |
| 48 | 0.503 | 0.487 | [0.452, 0.523] | [0.464, 0.503] | 0.500 | 0.501 |

</details>

<details>
<summary>Per-layer held-out AUROC — <code>response_final</code></summary>

| layer | validation AUROC | held-out AUROC | row-bootstrap 95% CI | template-bootstrap 95% CI | shuffled-label | random-direction |
|---|---|---|---|---|---|---|
| 0 | 0.500 | 0.500 | [0.500, 0.500] | [0.500, 0.500] | 0.500 | 0.500 |
| 1 | 0.500 | 0.513 | [0.477, 0.548] | [0.479, 0.545] | 0.491 | 0.495 |
| 2 | 0.507 | 0.502 | [0.466, 0.538] | [0.461, 0.540] | 0.509 | 0.500 |
| 3 | 0.495 | 0.479 | [0.442, 0.515] | [0.445, 0.511] | 0.519 | 0.496 |
| 4 | 0.500 | 0.478 | [0.442, 0.514] | [0.451, 0.508] | 0.512 | 0.500 |
| 5 | 0.494 | 0.476 | [0.440, 0.513] | [0.433, 0.519] | 0.495 | 0.496 |
| 6 | 0.505 | 0.489 | [0.453, 0.525] | [0.454, 0.529] | 0.482 | 0.495 |
| 7 | 0.507 | 0.504 | [0.468, 0.539] | [0.465, 0.546] | 0.494 | 0.508 |
| 8 | 0.496 | 0.503 | [0.467, 0.538] | [0.472, 0.539] | 0.487 | 0.507 |
| 9 | 0.494 | 0.515 | [0.478, 0.550] | [0.489, 0.547] | 0.500 | 0.499 |
| 10 | 0.472 | 0.521 | [0.486, 0.557] | [0.495, 0.549] | 0.479 | 0.500 |
| 11 | 0.482 | 0.520 | [0.484, 0.556] | [0.494, 0.545] | 0.520 | 0.512 |
| 12 | 0.491 | 0.508 | [0.471, 0.543] | [0.474, 0.538] | 0.492 | 0.484 |
| 13 | 0.504 | 0.490 | [0.454, 0.526] | [0.455, 0.521] | 0.497 | 0.500 |
| 14 | 0.489 | 0.489 | [0.452, 0.524] | [0.450, 0.531] | 0.507 | 0.478 |
| 15 | 0.496 | 0.503 | [0.467, 0.539] | [0.468, 0.547] | 0.503 | 0.492 |
| 16 | 0.489 | 0.493 | [0.457, 0.528] | [0.460, 0.533] | 0.492 | 0.498 |
| 17 | 0.476 | 0.492 | [0.456, 0.528] | [0.460, 0.535] | 0.495 | 0.499 |
| 18 | 0.488 | 0.505 | [0.468, 0.540] | [0.478, 0.536] | 0.485 | 0.515 |
| 19 | 0.492 | 0.502 | [0.466, 0.537] | [0.462, 0.537] | 0.484 | 0.505 |
| 20 | 0.499 | 0.507 | [0.471, 0.543] | [0.478, 0.536] | 0.477 | 0.507 |
| 21 | 0.494 | 0.500 | [0.464, 0.535] | [0.469, 0.533] | 0.492 | 0.496 |
| 22 | 0.502 | 0.486 | [0.451, 0.523] | [0.464, 0.511] | 0.494 | 0.490 |
| 23 | 0.501 | 0.495 | [0.460, 0.531] | [0.474, 0.518] | 0.500 | 0.507 |
| 24 | 0.491 | 0.482 | [0.446, 0.518] | [0.456, 0.510] | 0.491 | 0.492 |
| 25 | 0.503 | 0.491 | [0.455, 0.527] | [0.464, 0.520] | 0.511 | 0.489 |
| 26 | 0.491 | 0.501 | [0.465, 0.537] | [0.479, 0.528] | 0.506 | 0.495 |
| 27 | 0.475 | 0.506 | [0.471, 0.543] | [0.481, 0.541] | 0.486 | 0.485 |
| 28 | 0.481 | 0.501 | [0.465, 0.537] | [0.478, 0.528] | 0.493 | 0.493 |
| 29 | 0.482 | 0.497 | [0.461, 0.533] | [0.472, 0.522] | 0.517 | 0.503 |
| 30 | 0.495 | 0.492 | [0.456, 0.527] | [0.475, 0.511] | 0.531 | 0.503 |
| 31 | 0.487 | 0.501 | [0.466, 0.536] | [0.484, 0.524] | 0.504 | 0.496 |
| 32 | 0.489 | 0.497 | [0.462, 0.532] | [0.483, 0.516] | 0.489 | 0.497 |
| 33 | 0.510 | 0.510 | [0.474, 0.545] | [0.495, 0.527] | 0.523 | 0.507 |
| 34 | 0.501 | 0.518 | [0.481, 0.554] | [0.496, 0.541] | 0.520 | 0.485 |
| 35 | 0.513 | 0.516 | [0.481, 0.553] | [0.492, 0.545] | 0.506 | 0.508 |
| 36 | 0.520 | 0.513 | [0.477, 0.549] | [0.484, 0.539] | 0.517 | 0.501 |
| 37 | 0.505 | 0.505 | [0.469, 0.541] | [0.477, 0.534] | 0.459 | 0.492 |
| 38 | 0.498 | 0.501 | [0.465, 0.536] | [0.473, 0.529] | 0.499 | 0.510 |
| 39 | 0.512 | 0.502 | [0.466, 0.538] | [0.470, 0.532] | 0.504 | 0.504 |
| 40 | 0.502 | 0.517 | [0.480, 0.553] | [0.487, 0.546] | 0.505 | 0.512 |
| 41 | 0.507 | 0.516 | [0.480, 0.551] | [0.488, 0.546] | 0.490 | 0.492 |
| 42 | 0.497 | 0.511 | [0.475, 0.546] | [0.487, 0.536] | 0.514 | 0.498 |
| 43 | 0.496 | 0.511 | [0.475, 0.547] | [0.489, 0.539] | 0.540 | 0.496 |
| 44 | 0.500 | 0.510 | [0.473, 0.546] | [0.488, 0.536] | 0.517 | 0.505 |
| 45 | 0.490 | 0.509 | [0.472, 0.545] | [0.491, 0.532] | 0.500 | 0.512 |
| 46 | 0.491 | 0.509 | [0.473, 0.545] | [0.492, 0.528] | 0.504 | 0.500 |
| 47 | 0.488 | 0.505 | [0.470, 0.541] | [0.485, 0.530] | 0.499 | 0.493 |
| 48 | 0.493 | 0.507 | [0.471, 0.543] | [0.482, 0.530] | 0.490 | 0.506 |

</details>

## Files

- `inventory.json` — leak-free staging inventory (`inventory --leak-free`).
- `prompt_final/consistency_check.json`, `response_final/consistency_check.json` — the consistency
  report per cache (framing, prompt identity, per-layer AUROC with bootstrap CIs, LBR, verdict).
- `prompt_final/position_results.json`, `response_final/position_results.json` — the full sealed-path
  `train._run_locked_position` output the consistency report is derived from.

## Exact commands

```bash
export MODAL_PROFILE=<your-modal-profile>
# Collection (collector code and MODAL_GPU unchanged; shard-aware resume)
python -m experiments.exp4_collection submit main --leak-free
python -m experiments.exp4_collection submit lbr --leak-free
# Download + finalize (CPU)
mkdir -p experiments/exp4_collection/local/staging_leak_free
modal volume get vmp-activations leak_free/collect_main experiments/exp4_collection/local/staging_leak_free/collect_main
modal volume get vmp-activations leak_free/collect_lbr experiments/exp4_collection/local/staging_leak_free/collect_lbr
python3 -m experiments.exp4_collection inventory --staging-dir experiments/exp4_collection/local/staging_leak_free --leak-free
python3 -m experiments.exp4_collection finalize --staging-dir experiments/exp4_collection/local/staging_leak_free \
  --output-dir experiments/exp4_collection/local/final_leak_free --leak-free
# Consistency check (CPU), one run per finalized cache
for pos in prompt_final response_final; do
  python3 -m experiments.exp4_probes.leak_free_consistency \
    --cache experiments/exp4_collection/local/final_leak_free/$pos.npz \
    --metadata experiments/exp4_collection/local/final_leak_free/transcripts.jsonl \
    --output-dir experiments/exp4_analysis/local/results_leak_free/$pos
done
```

`experiments/exp4_analysis/derive.py` (GPU, `receipt_final`) was deliberately not run: it is not
wired to the leak-free variant so no derived measurement inherits the tautology silently.
