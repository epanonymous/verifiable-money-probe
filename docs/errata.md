# Errata and seal integrity

Corrections that would otherwise require editing a sealed artifact live here.

The Wave 2 evidence bundle under
[`experiments/exp4_analysis/results/wave2/`](../experiments/exp4_analysis/results/wave2/)
is sealed: [`evidence_manifest.json`](../experiments/exp4_analysis/results/wave2/evidence_manifest.json)
pins the path, byte size, and SHA-256 of every other file in that directory
(25 artifacts; `manifest_scope` excludes only the manifest's own recursive
self-hash). Nothing in that directory may be edited in place. Wording,
scope, and interpretation corrections are recorded here instead, and the seal
is checked by
[`experiments/exp4_analysis/tests/test_evidence_manifest.py`](../experiments/exp4_analysis/tests/test_evidence_manifest.py),
which runs in the CPU suite.

Sealed numbers are never restated or overridden here. Every entry below is a
scope/wording correction only.

## E1 — the low-base-rate cache is same-generator, not independent

*Recorded 2026-09-02*

**Sealed text as written.** `results/wave2/README.md` calls the 10 REAL :
1,000 SHAM empirical low-base-rate (LBR) cache "**independent**", both in the
class-contract paragraph and in the fixed-threshold sentence under *Canonical
gates and results*. The sealed JSON uses the same word in identifiers:
`independent_lbr` in `positions/*/position_results.json` and
`analysis/checkpoint_results.json`, `independent_low_base_rate_cache` in the
Wave 4 text baseline, and the `independent_lbr_REAL_vs_SHAM.jsonl` score
filenames.

**What is actually true.** The cache is *separately generated* — a distinct
20-shard collection run that never trained or validated a probe — but it comes
from the **same prompt generator and the same subject model**
(`Qwen/Qwen3-30B-A3B-Instruct-2507`) and preserves the same literal verifier
strings. It is therefore a **separately generated, same-generator** sample.
**It is not an independent-distribution, out-of-distribution, or paraphrase
test**, and it does not provide independent evidence for a latent
representation. The zero-error LBR result inherits the verifier-string leak
described in `docs/run-v1.md`; a same-generator cache cannot test
generalization past that leak.

**Scope of the correction.** Wording and interpretation only. No count,
metric, threshold, hash, or file changes: the cache is still 10 REAL : 1,000
SHAM (10/1,010 = 0.990% REAL prevalence), and the sealed result is still
TP=10, FN=0, TN=1,000, FP=0 at threshold 0.5 for both positions.

**Where the corrected wording already lives.** `docs/writeup/research-task.md`
(§3.2, §4.2), `docs/run-v1.md`, `experiments/exp4_probes/README.md`, and the
headline figure's accessible `<desc>` in
`docs/writeup/assets/wave4-headline-results.svg` and its generator
`experiments/exp4_analysis/generate_headline_figure.py`.

**Deliberately not changed.** The sealed `results/wave2/README.md`, and the
`independent_lbr` / `independent_low_base_rate_cache` JSON keys and score
filenames. Those are sealed identifiers; renaming them would break the seal
and every consumer that reads them. Read them as "separately generated,
same-generator LBR", per this entry.

## Seal-integrity ledger

Deviations between the bundle as it stands on `main` and the hashes pinned in
`evidence_manifest.json`. The manifest is the original seal and is **not**
rewritten; drift is recorded here instead.

| Artifact | Manifest-pinned | Current on `main` | Introduced by | Numbers affected |
|---|---|---|---|---|
| `README.md` | 8,808 B · `0637290aa62c31be692619b142299a851b31163f99395fa93209706d41634818` | 9,327 B · `747c4998c52dcf74bd01687a50d54dc4d727e3b26ef63a533aa93331222d7103` | banner prepend `a2463c5` | none |
| all 24 others | — | matches manifest | — | none |

`a2463c5` prepended a "Run v1 interpretation (do not rewrite these sealed
numbers)" banner to the bundle README without versioning the seal. That is a
pre-existing deviation, recorded here rather than laundered: the added text is
an interpretive warning, and every table, count, and metric in that README is
byte-unchanged from the sealed original. The original bytes hash to the
manifest-pinned `0637290a…4818`.

`test_evidence_manifest.py` verifies all 25 artifacts, pins this one accepted
deviation to its exact current hash, and fails on any further edit to the
bundle — including any new file added under the bundle root.
