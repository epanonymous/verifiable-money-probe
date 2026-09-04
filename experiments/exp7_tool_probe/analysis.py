"""Exp 7 v0 (#31) — CPU analysis of the Modal shards.

Inputs (downloaded from the vmp-activations volume, exp7/v0/):
    <acts>/main/t??.npz   one shard per template, both worlds
    <acts>/auth/*.npz     one shard per authenticity row
    data/v0/              frozen inputs (rows, readouts, captures, auth split manifest)

Deliverables written to --out:
    results.json       every number below, plus a provenance block linking them to the shards
    summary.md         the write-up section (numbers, honest caveats, pre-registration deviations)
    blind_p0.json      the blind P0 gate (post hoc for v0; see blind_p0.py)
    fig_layers.png     per-layer AUROC curves (P0 leak check, P1, V5 side probe)
    fig_decisions.png  SPEND rate by world + within-world probe score vs SPEND rate
    transcripts.jsonl  every rollout text and decision (archival)

Probe = standardized logistic regression (C=0.1, liblinear), one per layer, the Exp 4
recipe. Evaluation splits never cut a template (main) or a real/forged pair (auth):
    heldout   train on the 38 train templates, score the 10 held-out paraphrase templates
    LOTO      leave-one-template-out over all 48, out-of-fold scores pooled
Layer selection for headline numbers uses GroupKFold(5) on the train templates only.

V5 split: the primary held-out split is stratified by capture kind
(data/v0/auth_split.json). The superseded index-tail split baked into the shards'
``split`` field (24/24 native_balance held out) is evaluated too and reported next
to it so the change is visible.

Everything random (probe solver, bootstrap, blind-gate shuffles) is seeded from
config.SEED; two runs on the same shards produce byte-identical results.json.

Provenance preflight: before a single shard is read, the committed
data/<ver>/shards.sha256 is checked against the mirror — every listed shard
(48 main + 120 auth for v0) and every committed input must be present and
match, or the run raises RuntimeError. ``--skip-provenance-check`` bypasses
this for deliberately partial runs and is recorded in results.json.

Independent-model blind judge: the network call happens only in
``blind_p0.py --llm``; it writes ``llm_judge_record.json``. This module never
calls an API — pass ``--llm-record <that file>`` to fold the recorded replies
into the gate deterministically. Without it the gate verdict is INCOMPLETE
unless ``--allow-no-llm`` is passed (recorded).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from . import blind_p0 as blind
from . import provenance as prov
from .config import N_BOOT, PROBE_C, PROBE_MAX_ITER, SEED, config_hash, seed_everything

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_DATA = HERE / "data" / "v0"
C = PROBE_C
MAX_ITER = PROBE_MAX_ITER
N_JOBS = int(os.environ.get("EXP7_JOBS", "3"))  # per-layer fits are independent; keep the host responsive
P0_ALARM_AUROC = 0.65


# ----------------------------------------------------------------------------- probes
def fit_scores(X_tr, y_tr, X_te):
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=C, max_iter=MAX_ITER, random_state=SEED, solver="liblinear")
    clf.fit(scaler.transform(X_tr), y_tr)
    return clf.decision_function(scaler.transform(X_te))


def safe_auroc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def boot_ci(y, s, groups, n=None, seed=SEED):
    """Group-level bootstrap CI for AUROC (resample templates/pairs with replacement).

    Resamples with one class only are dropped, so the CI is conditional on both
    classes being present (matters only when one class is rare).
    """

    n = N_BOOT if n is None else n
    rng = np.random.default_rng(seed)
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        a = safe_auroc(y[idx], s[idx])
        if not np.isnan(a):
            vals.append(a)
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def loto_scores(X, y, groups, layer):
    """Leave-one-group-out out-of-fold decision scores at one layer."""

    out = np.zeros(len(y))
    for g in np.unique(groups):
        te = groups == g
        out[te] = fit_scores(X[~te, layer, :], y[~te], X[te, layer, :])
    return out


def layer_curve_loto(X, y, groups):
    n_layers = X.shape[1]
    cols = Parallel(n_jobs=N_JOBS)(delayed(loto_scores)(X, y, groups, layer) for layer in range(n_layers))
    scores = np.stack(cols, axis=1)
    return [safe_auroc(y, scores[:, l]) for l in range(n_layers)], scores


def layer_curve_heldout(X, y, groups, is_train):
    n_layers = X.shape[1]
    if (~is_train).sum() == 0:  # partial run with no held-out templates yet
        return [float("nan")] * n_layers, [float("nan")] * n_layers
    ss = Parallel(n_jobs=N_JOBS)(
        delayed(fit_scores)(X[is_train, layer, :], y[is_train], X[~is_train, layer, :]) for layer in range(n_layers)
    )
    aur = [safe_auroc(y[~is_train], s) for s in ss]
    acc = [float(np.mean((s > 0).astype(int) == y[~is_train])) for s in ss]
    return aur, acc


def select_layer(X, y, groups, is_train, n_splits=5):
    """Pick the layer by GroupKFold CV inside the train templates only."""

    Xtr, ytr, gtr = X[is_train], y[is_train], groups[is_train]
    n_layers = X.shape[1]
    n_groups = len(np.unique(gtr))
    if n_groups < 2:
        raise RuntimeError("layer selection needs at least two training groups")
    folds = list(GroupKFold(n_splits=min(n_splits, n_groups)).split(Xtr, ytr, gtr))

    def one(layer):
        s = np.zeros(len(ytr))
        for tr, te in folds:
            s[te] = fit_scores(Xtr[tr, layer, :], ytr[tr], Xtr[te, layer, :])
        return safe_auroc(ytr, s)

    cv = np.array(Parallel(n_jobs=N_JOBS)(delayed(one)(layer) for layer in range(n_layers)))
    # ties -> earliest layer (cheapest/most conservative)
    best = int(np.nanargmax(cv))
    return best, [float(v) for v in cv]


# ----------------------------------------------------------------------------- loading
def _p0_pair_stats(r0: np.ndarray, s0: np.ndarray) -> dict:
    """How far apart the two worlds' P0 activations are, globally and per layer."""

    diff = np.abs(r0 - s0)
    layer_norm_rel = np.linalg.norm(r0 - s0, axis=1) / (np.linalg.norm(r0, axis=1) + 1e-9)
    cos = np.sum(r0 * s0, axis=1) / (np.linalg.norm(r0, axis=1) * np.linalg.norm(s0, axis=1) + 1e-9)
    return {
        "p0_pair_maxdiff": float(diff.max()),
        "p0_pair_reldiff": float(diff.max() / np.abs(r0).max()),
        "p0_pair_layer_relnorm_max": float(layer_norm_rel.max()),
        "p0_pair_layer_relnorm_argmax": int(layer_norm_rel.argmax()),
        "p0_pair_cos_min": float(cos.min()),
    }


def load_main(acts_dir):
    shards = sorted(glob.glob(os.path.join(acts_dir, "main", "t*.npz")))
    if not shards:
        raise FileNotFoundError(f"no main shards under {acts_dir}")
    rows = []
    for path in shards:
        with np.load(path) as z:
            tid = int(z["template_id"])
            split = str(z["split"])
            source = str(z["tool_call_source"])
            stats = _p0_pair_stats(z["real_p0"].astype(np.float32), z["sham_p0"].astype(np.float32))
            has_stamp = "config_hash" in z.files
            for world in ("REAL", "SHAM"):
                w = world.lower()
                rows.append(
                    {
                        "template_id": tid,
                        "split": split,
                        "world": world,
                        "label": int(z[f"{w}_label"]),
                        "p0": z[f"{w}_p0"].astype(np.float32),
                        "p1": z[f"{w}_p1"].astype(np.float32),
                        "resp": z[f"{w}_resp"].astype(np.float32),
                        "decisions": [str(d) for d in z[f"{w}_decisions"]],
                        "texts": [str(t) for t in z[f"{w}_texts"]],
                        "tool_text": str(z[f"{w}_tool_text"]),
                        "p1_spread": float(z[f"{w}_p1_spread"]),
                        "p1_len": int(z[f"{w}_p1_len"]),
                        "tool_call_source": source,
                        "stage_a_text": str(z["stage_a_text"]),
                        "p0_len": int(z["p0_len"]),
                        "block": int(z["block"]),
                        "model": str(z["model"]),
                        "gpu": str(z["gpu"]),
                        "has_config_hash": has_stamp,
                        **stats,
                    }
                )
    return rows


def load_auth(acts_dir, split_override: dict[int, str] | None = None):
    """Auth rows; ``split_shard`` is the split baked into the shard, ``split`` the one used."""

    shards = sorted(glob.glob(os.path.join(acts_dir, "auth", "auth_*.npz")))
    if os.path.isdir(os.path.join(acts_dir, "auth")) and not shards:
        print(f"[analysis] WARNING: {acts_dir}/auth exists but holds no auth_*.npz shards")
    rows = []
    for path in shards:
        with np.load(path) as z:
            tid = int(z["template_id"])
            split_shard = str(z["split"])
            rows.append(
                {
                    "row_id": str(z["row_id"]),
                    "kind": str(z["kind"]),
                    "method": str(z["method"]),
                    "capture_kind": str(z["capture_kind"]),
                    "template_id": tid,
                    "split_shard": split_shard,
                    "split": split_override[tid] if split_override is not None else split_shard,
                    "label": int(z["label"]),
                    "prompt": z["prompt_final"][0].astype(np.float32),
                    "resp": z["response_final"][0].astype(np.float32),
                    "verdict": str(z["verdicts"][0]),
                    "text": str(z["texts"][0]),
                }
            )
    return rows


def load_auth_split(data_dir) -> dict[int, str] | None:
    path = Path(data_dir) / "auth_split.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text())
    return {i: s for i, s in enumerate(manifest["split"])}


# ----------------------------------------------------------------------------- main analysis
def analyse_main(rows):
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["template_id"] for r in rows])
    is_train = np.array([r["split"] == "train" for r in rows])
    X0 = np.stack([r["p0"] for r in rows])  # [n, L+1, d]
    X1 = np.stack([r["p1"] for r in rows])
    n_layers = X1.shape[1]

    out = {}
    out["n_rows"] = int(len(rows))
    out["n_templates"] = int(len(np.unique(groups)))
    out["n_layers_incl_embed"] = int(n_layers)
    out["d_model"] = int(X1.shape[2])
    out["model"] = rows[0]["model"]
    out["gpu"] = rows[0]["gpu"]
    out["block"] = rows[0]["block"]
    out["tool_call_source_rate"] = {
        s: float(np.mean([r["tool_call_source"] == s for r in rows]))
        for s in ("model", "scripted_fallback")
    }
    out["stage_a_unique_texts"] = sorted({r["stage_a_text"] for r in rows})[:5]
    out["tool_text"] = {w: next(r["tool_text"] for r in rows if r["world"] == w) for w in ("REAL", "SHAM")}

    # --- P0 activation sanity check: byte-identical prefix -> activations equal up to numerics
    out["p0_pair_maxdiff_max"] = float(max(r["p0_pair_maxdiff"] for r in rows))
    out["p0_pair_reldiff_max"] = float(max(r["p0_pair_reldiff"] for r in rows))
    out["p0_pair_layer_relnorm_max"] = float(max(r["p0_pair_layer_relnorm_max"] for r in rows))
    out["p0_pair_layer_relnorm_argmax_layers"] = sorted({int(r["p0_pair_layer_relnorm_argmax"]) for r in rows})
    out["p0_pair_cos_min"] = float(min(r["p0_pair_cos_min"] for r in rows))
    out["p0_activation_check_note"] = (
        "P0 token ids are shared across worlds by construction; the activations were captured "
        "from full (P1 + rollout) sequences batched per world, so they differ by batched bf16 "
        "kernel numerics (MoE routing can flip). No prefix-only forward was run to bound this "
        "and no tolerance was pre-declared; the operative leak check is the P0 probe below."
    )
    out["n_train_templates"] = int(len(np.unique(groups[is_train])))
    out["n_heldout_templates"] = int(len(np.unique(groups[~is_train])))
    out["p1_rollout_spread_max"] = float(max(r["p1_spread"] for r in rows))
    p0_loto, _ = layer_curve_loto(X0, y, groups)
    p0_held, _ = layer_curve_heldout(X0, y, groups, is_train)
    out["p0"] = {
        "loto_auroc_by_layer": p0_loto,
        "heldout_auroc_by_layer": p0_held,
        "loto_auroc_max": float(np.nanmax(p0_loto)),
        "heldout_auroc_max": float(np.nanmax(p0_held)),
        "alarm_auroc": P0_ALARM_AUROC,
    }

    # --- P1: real-vs-sham direction after the tool result lands
    sel, cv_curve = select_layer(X1, y, groups, is_train)
    p1_held, p1_held_acc = layer_curve_heldout(X1, y, groups, is_train)
    p1_loto, p1_loto_scores = layer_curve_loto(X1, y, groups)
    has_held = (~is_train).sum() > 0
    s_held = fit_scores(X1[is_train, sel, :], y[is_train], X1[~is_train, sel, :]) if has_held else np.zeros(0)
    out["p1"] = {
        "selected_layer": sel,
        "selected_layer_rule": "argmax GroupKFold(5) AUROC on train templates only",
        "train_cv_auroc_by_layer": cv_curve,
        "heldout_auroc_by_layer": p1_held,
        "heldout_acc_by_layer": p1_held_acc,
        "loto_auroc_by_layer": p1_loto,
        "heldout_auroc_at_selected": float(p1_held[sel]),
        "heldout_acc_at_selected": float(p1_held_acc[sel]),
        "heldout_auroc_ci_at_selected": boot_ci(y[~is_train], s_held, groups[~is_train]) if has_held else [float("nan")] * 2,
        "loto_auroc_at_selected": float(p1_loto[sel]),
        "loto_auroc_ci_at_selected": boot_ci(y, p1_loto_scores[:, sel], groups),
        "n_heldout_rows": int((~is_train).sum()),
    }

    # --- text baseline: the tool JSON differs only in "balance"; a one-line rule is perfect
    out["text_baseline"] = {
        "rule": "balance != '0.000000' -> REAL",
        "world_auroc": 1.0,
        "note": (
            "REAL and SHAM differ only in the balance field by construction, so raw tool "
            "text separates the worlds perfectly; the probe cannot beat it on world identity. "
            "The informative comparisons are generalization to held-out paraphrases and "
            "within-world prediction of the decision, where the tool text is constant."
        ),
    }

    # --- decisions
    dec = []
    for i, r in enumerate(rows):
        for k, d in enumerate(r["decisions"]):
            dec.append({"row": i, "world": r["world"], "split": r["split"], "template_id": r["template_id"], "k": k, "decision": d})

    def rate(world, what):
        ds = [d["decision"] for d in dec if d["world"] == world]
        return float(np.mean([d == what for d in ds])) if ds else float("nan")

    out["decisions"] = {
        "n_rollouts_total": len(dec),
        "spend_rate": {w: rate(w, "SPEND") for w in ("REAL", "SHAM")},
        "hold_rate": {w: rate(w, "HOLD") for w in ("REAL", "SHAM")},
        "ambiguous_rate": {w: rate(w, "AMBIGUOUS") for w in ("REAL", "SHAM")},
        "row_spend_rate": [
            float(np.mean([d == "SPEND" for d in r["decisions"] if d != "AMBIGUOUS"] or [np.nan]))
            for r in rows
        ],
    }

    # --- does the probe predict the decision better than the raw tool text?
    # per-rollout target: SPEND=1 / HOLD=0 (ambiguous dropped)
    keep = [d for d in dec if d["decision"] in ("SPEND", "HOLD")]
    dy = np.array([1 if d["decision"] == "SPEND" else 0 for d in keep])
    d_world = np.array([1 if d["world"] == "REAL" else 0 for d in keep])
    d_row = np.array([d["row"] for d in keep])
    d_group = np.array([d["template_id"] for d in keep])
    p1_oof = p1_loto_scores[:, sel]  # one out-of-fold world score per row
    d_p1 = p1_oof[d_row]
    # response_final: world probe trained LOTO on per-rollout post-decision activations
    Xr = np.concatenate([r["resp"] for r in rows])  # [n_rows*n_roll, L+1, d]
    yr = np.concatenate([[r["label"]] * len(r["decisions"]) for r in rows])
    gr = np.concatenate([[r["template_id"]] * len(r["decisions"]) for r in rows])
    rollout_index = np.concatenate([[(i, k) for k in range(len(r["decisions"]))] for i, r in enumerate(rows)])
    resp_oof = loto_scores(Xr, yr, gr, sel)
    resp_lookup = {tuple(map(int, rollout_index[j])): resp_oof[j] for j in range(len(resp_oof))}
    d_resp = np.array([resp_lookup[(d["row"], d["k"])] for d in keep])

    def within(world, s):
        m = d_world == (1 if world == "REAL" else 0)
        return {
            "auroc": safe_auroc(dy[m], s[m]),
            "ci": boot_ci(dy[m], s[m], d_group[m]),
            "n": int(m.sum()),
            "n_spend": int(dy[m].sum()),
            "n_unique_scores": int(len(np.unique(s[m]))),
        }

    out["decision_prediction"] = {
        "target": "per-rollout SPEND(1)/HOLD(0); AMBIGUOUS dropped",
        "n_rollouts_used": int(len(keep)),
        "text_baseline_world_label": {
            "auroc": safe_auroc(dy, d_world),
            "ci": boot_ci(dy, d_world, d_group),
        },
        "p1_probe_oof_score": {
            "auroc_all": safe_auroc(dy, d_p1),
            "ci_all": boot_ci(dy, d_p1, d_group),
            "within_REAL": within("REAL", d_p1),
            "within_SHAM": within("SHAM", d_p1),
            "note": "P1 is a prefix position: one score per row is repeated across its rollouts (n_unique_scores)",
        },
        "response_final_probe_oof_score": {
            "auroc_all": safe_auroc(dy, d_resp),
            "ci_all": boot_ci(dy, d_resp, d_group),
            "within_REAL": within("REAL", d_resp),
            "within_SHAM": within("SHAM", d_resp),
            "note": "post-decision position: predictivity here can be the decision itself, not belief",
        },
        "resp_world_auroc_loto_at_selected": safe_auroc(yr, resp_oof),
    }
    # row-level within-world: probe score vs spend rate (Spearman)
    row_rate = np.array(out["decisions"]["row_spend_rate"])
    sp = {}
    for world in ("REAL", "SHAM"):
        m = np.array([r["world"] == world for r in rows]) & ~np.isnan(row_rate)
        if m.sum() > 3 and np.nanstd(row_rate[m]) > 0:
            rho, p = spearmanr(p1_oof[m], row_rate[m])
            sp[world] = {"spearman_rho": float(rho), "p": float(p), "n_rows": int(m.sum())}
        else:
            sp[world] = {"spearman_rho": float("nan"), "p": float("nan"), "n_rows": int(m.sum()), "note": "no variance in spend rate"}
    out["decision_prediction"]["within_world_row_level_p1_vs_spend_rate"] = sp
    return out, {"p1_oof": p1_oof, "row_rate": row_rate, "worlds": [r["world"] for r in rows]}


# ----------------------------------------------------------------------------- V5 side probe
def _split_counts_by_kind(rows, split_key):
    out = {"train": {}, "heldout": {}}
    seen = set()
    for r in rows:
        if r["template_id"] in seen:
            continue
        seen.add(r["template_id"])
        out[r[split_key]][r["capture_kind"]] = out[r[split_key]].get(r["capture_kind"], 0) + 1
    return {s: dict(sorted(c.items())) for s, c in out.items()}


def analyse_auth(rows, split_key="split", *, lopo_cache=None):
    """V5 numbers under one split. ``lopo_cache`` (split-independent curves) can be shared."""

    if not rows:
        return {"n_rows": 0, "note": "no auth shards"}
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["template_id"] for r in rows])
    is_train = np.array([r[split_key] == "train" for r in rows])
    kinds = np.array([r["capture_kind"] for r in rows])
    out = {
        "n_rows": int(len(rows)),
        "n_pairs": int(len(np.unique(groups))),
        "split_key": split_key,
        "split_counts_pairs_by_kind": _split_counts_by_kind(rows, split_key),
        "n_heldout_rows": int((~is_train).sum()),
    }
    for name, X in (("prompt_final", np.stack([r["prompt"] for r in rows])), ("response_final", np.stack([r["resp"] for r in rows]))):
        if lopo_cache is not None and name in lopo_cache:
            lopo, lopo_scores = lopo_cache[name]
        else:
            lopo, lopo_scores = layer_curve_loto(X, y, groups)
            if lopo_cache is not None:
                lopo_cache[name] = (lopo, lopo_scores)
        sel, cv = select_layer(X, y, groups, is_train)
        held, held_acc = layer_curve_heldout(X, y, groups, is_train)
        s_held = fit_scores(X[is_train, sel, :], y[is_train], X[~is_train, sel, :])
        by_kind = {}
        for k in sorted(set(kinds)):
            m = kinds == k
            by_kind[k] = {"n": int(m.sum()), "lopo_auroc_at_selected": safe_auroc(y[m], lopo_scores[m, sel])}
        held_by_kind = {}
        for k in sorted(set(kinds[~is_train])):
            m = (kinds == k)[~is_train]
            held_by_kind[k] = {"n": int(m.sum()), "heldout_auroc_at_selected": safe_auroc(y[~is_train][m], s_held[m])}
        out[name] = {
            "selected_layer": sel,
            "train_cv_auroc_by_layer": cv,
            "heldout_auroc_by_layer": held,
            "lopo_auroc_by_layer": lopo,
            "heldout_auroc_at_selected": float(held[sel]),
            "heldout_acc_at_selected": float(held_acc[sel]),
            "heldout_auroc_ci_at_selected": boot_ci(y[~is_train], s_held, groups[~is_train]),
            "lopo_auroc_at_selected": float(lopo[sel]),
            "lopo_auroc_ci_at_selected": boot_ci(y, lopo_scores[:, sel], groups),
            "lopo_auroc_max_any_layer": float(np.nanmax(lopo)),
            "by_capture_kind": by_kind,
            "heldout_by_capture_kind": held_by_kind,
        }
    verdict_map = {"AUTHENTIC": 1, "FABRICATED": 0}
    v = [r["verdict"] for r in rows]
    answered = [(verdict_map[a], lab) for a, lab in zip(v, y) if a in verdict_map]
    out["verbal"] = {
        "ambiguous_rate": float(np.mean([a not in verdict_map for a in v])),
        "accuracy_on_answered": float(np.mean([a == lab for a, lab in answered])) if answered else float("nan"),
        "n_answered": len(answered),
        "authentic_rate_real": float(np.mean([a == "AUTHENTIC" for a, lab in zip(v, y) if lab == 1])),
        "authentic_rate_forged": float(np.mean([a == "AUTHENTIC" for a, lab in zip(v, y) if lab == 0])),
    }
    return out


def analyse_auth_both(rows, *, has_override: bool):
    """Primary (stratified) numbers plus the superseded tail split, sharing the LOPO curves."""

    if not rows:
        return {"n_rows": 0, "note": "no auth shards"}, None
    cache: dict = {}
    primary = analyse_auth(rows, "split", lopo_cache=cache)
    primary["split_scheme"] = "stratified_by_capture_kind (data/v0/auth_split.json)" if has_override else "as stored in shards"
    tail = None
    if has_override and any(r["split"] != r["split_shard"] for r in rows):
        tail = analyse_auth(rows, "split_shard", lopo_cache=cache)
        tail["split_scheme"] = "index tail as stored in the v0 shards (superseded: one capture kind held out)"
        for name in ("prompt_final", "response_final"):  # curves are identical by construction; keep the file small
            tail[name] = {k: v for k, v in tail[name].items() if k != "lopo_auroc_by_layer"}
    return primary, tail


# ----------------------------------------------------------------------------- provenance
def preflight_provenance(acts_dir, data_dir) -> dict:
    """Refuse to analyse an incomplete or altered mirror (RuntimeError names what is wrong)."""

    data_dir = Path(data_dir)
    manifest_path = data_dir / prov.SHA_MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError(
            f"provenance preflight failed: no {prov.SHA_MANIFEST_NAME} under {data_dir}; write it with "
            "verify_provenance --acts <dir> --write, or pass --skip-provenance-check for a deliberately partial run (recorded)"
        )
    data_manifest_path = data_dir / "manifest.json"
    if not data_manifest_path.is_file():
        raise RuntimeError(f"provenance preflight failed: {data_manifest_path} is missing")
    data_manifest = json.loads(data_manifest_path.read_text())
    try:
        expected = {"main": int(data_manifest["n_templates"]), "auth": int(data_manifest["n_auth_rows"])}
    except KeyError as exc:
        raise RuntimeError(f"provenance preflight failed: data manifest lacks {exc} (expected shard inventory)") from None
    records = prov.read_sha256_manifest(manifest_path)
    summary = prov.check_inventory(records, data_dir=data_dir, acts_dir=acts_dir, expected_shards=expected)
    summary["manifest"] = prov.SHA_MANIFEST_NAME
    summary["expected_shards"] = expected
    return summary


def provenance_block(acts_dir, data_dir, main_rows, auth_rows):
    manifest_path = Path(data_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    main_records = prov.collect_records(Path(acts_dir) / "main", prefix="acts/main") if (Path(acts_dir) / "main").is_dir() else []
    auth_records = prov.collect_records(Path(acts_dir) / "auth", prefix="acts/auth") if (Path(acts_dir) / "auth").is_dir() else []
    versions = prov.runtime_versions()
    return {
        "collection": manifest.get("collection", {}),
        "amendments": manifest.get("amendments", {}),
        "block": manifest.get("block"),
        "data_files_sha256": manifest.get("files", {}),
        "model": main_rows[0]["model"] if main_rows else None,
        "gpu": main_rows[0]["gpu"] if main_rows else None,
        "n_main_shards": len(main_records),
        "n_auth_shards": len(auth_records),
        "shards_aggregate_sha256": {
            "main": prov.aggregate_sha256(main_records),
            "auth": prov.aggregate_sha256(auth_records),
            "all": prov.aggregate_sha256(main_records + auth_records),
        },
        "shards_carry_config_hash": bool(main_rows) and all(r["has_config_hash"] for r in main_rows),
        "preflight": None,  # filled by main(): the inventory check that ran before the shards were read
        "analysis": {
            "config_hash": config_hash(),
            "seed": SEED,
            "probe": {"model": "StandardScaler + LogisticRegression", "C": C, "max_iter": MAX_ITER, "solver": "liblinear"},
            "n_boot": N_BOOT,
            "versions": {k: versions.get(k) for k in ("python", "numpy", "sklearn", "scipy")},
        },
    }


# ----------------------------------------------------------------------------- figures
def figures(main, aux, auth, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = np.arange(main["n_layers_incl_embed"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(layers, main["p0"]["loto_auroc_by_layer"], label="P0 (pre-tool-result, byte-identical) LOTO", color="gray")
    ax.plot(layers, main["p1"]["loto_auroc_by_layer"], label=f"P1 (post-tool-result) LOTO, {main['n_templates']} templates", color="C0")
    ax.plot(layers, main["p1"]["heldout_auroc_by_layer"], label=f"P1 held-out paraphrases ({main['n_heldout_templates']} templates)", color="C0", ls="--")
    if auth.get("prompt_final"):
        ax.plot(layers, auth["prompt_final"]["lopo_auroc_by_layer"], label="V5 side probe: real vs forged RPC text (LOPO)", color="C3")
        ax.plot(layers, auth["prompt_final"]["heldout_auroc_by_layer"], label="V5 held-out pairs (stratified by kind)", color="C3", ls="--", alpha=0.7)
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.axvline(main["p1"]["selected_layer"], color="C0", lw=0.6, alpha=0.5)
    ax.set_xlabel("layer (0 = embeddings)")
    ax.set_ylabel("AUROC (REAL vs SHAM)")
    ax.set_ylim(0.3, 1.02)
    ax.set_title("Exp 7 v0 — tool-grounded belief probe, Qwen3-30B-A3B")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_layers.png"), dpi=150, metadata={"Software": None})
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    d = main["decisions"]
    worlds = ["REAL", "SHAM"]
    axes[0].bar(worlds, [d["spend_rate"][w] for w in worlds], color=["C2", "C1"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("SPEND rate over rollouts")
    axes[0].set_title("Stated decision by world")
    for w, c in zip(worlds, ["C2", "C1"]):
        m = np.array([x == w for x in aux["worlds"]])
        axes[1].scatter(aux["p1_oof"][m], aux["row_rate"][m], label=w, color=c, alpha=0.7)
    axes[1].set_xlabel("P1 probe score (out-of-fold, selected layer)")
    axes[1].set_ylabel("row SPEND rate")
    axes[1].set_title("Within-world: probe score vs decision")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_decisions.png"), dpi=150, metadata={"Software": None})
    plt.close(fig)


# ----------------------------------------------------------------------------- write-up
def fmt(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def ci(c):
    return f"[{fmt(c[0])}, {fmt(c[1])}]"


def p0_verdict(main):
    best = max(main["p0"]["loto_auroc_max"], main["p0"]["heldout_auroc_max"])
    if best <= P0_ALARM_AUROC:
        return "(a leak would show here as AUROC well above chance; it does not)."
    return "(**ALARM**: above chance on a byte-identical prefix — treat every downstream number as suspect until explained)."


def within_world_line(dp):
    r, s_ = dp["p1_probe_oof_score"]["within_REAL"], dp["p1_probe_oof_score"]["within_SHAM"]
    if np.isnan(r["auroc"]) and np.isnan(s_["auroc"]):
        return (
            f"- Within-world (tool text constant): undefined — every REAL rollout said SPEND ({r['n_spend']}/{r['n']}) and every "
            f"SHAM rollout said HOLD ({s_['n_spend']}/{s_['n']} SPEND). In v0 the decision is fully determined by the tool text, "
            "so no within-world signal exists for the probe to beat."
        )
    return (
        f"- Within-world (tool text constant): P1 probe vs decision AUROC REAL {fmt(r['auroc'])} {ci(r['ci'])} "
        f"(n={r['n']} rollouts, {r['n_spend']} SPEND, but only {r['n_unique_scores']} distinct P1 scores — one per row — so the "
        f"row-level Spearman below is the primary within-world number); SHAM {fmt(s_['auroc'])} {ci(s_['ci'])} (n={s_['n']}, {s_['n_spend']} SPEND)."
    )


def v5_lines(auth, tail):
    a = auth["prompt_final"]
    v = auth["verbal"]
    counts = auth["split_counts_pairs_by_kind"]
    lo, hi = a["lopo_auroc_ci_at_selected"]
    lopo_read = "its CI excludes 0.5, so the all-kind exploratory number is weak but not null" if lo > 0.5 else "its CI includes 0.5"
    hlo, hhi = a["heldout_auroc_ci_at_selected"]
    held_read = "CI excludes 0.5" if hlo > 0.5 else "CI includes 0.5"
    lines = [
        "### V5 side probe — real Base JSON-RPC responses vs well-forged fakes (text only)",
        f"- {auth['n_rows']} rows ({auth['n_pairs']} real/forged pairs). Held-out split stratified by capture kind, pairs never cut "
        f"(held-out pairs by kind: {', '.join(f'{k} {n}' for k, n in counts['heldout'].items())}; train: "
        f"{', '.join(f'{k} {n}' for k, n in counts['train'].items())}).",
        f"- prompt_final probe, selected layer {a['selected_layer']} (train-only CV): **held-out pairs AUROC {fmt(a['heldout_auroc_at_selected'])} "
        f"{ci(a['heldout_auroc_ci_at_selected'])}** ({held_read}), accuracy {fmt(a['heldout_acc_at_selected'])}; "
        f"leave-one-pair-out over all {auth['n_pairs']} pairs (exploratory, layer chosen on the same pairs' train CV) AUROC "
        f"{fmt(a['lopo_auroc_at_selected'])} {ci(a['lopo_auroc_ci_at_selected'])} — {lopo_read} (best any layer {fmt(a['lopo_auroc_max_any_layer'])}, uncorrected).",
        "- Held-out by capture kind (AUROC at selected layer): "
        + ", ".join(f"{k} {fmt(v_['heldout_auroc_at_selected'], 2)} (n={v_['n']})" for k, v_ in a["heldout_by_capture_kind"].items()) + ".",
        "- LOPO by capture kind (AUROC at selected layer): "
        + ", ".join(f"{k} {fmt(v_['lopo_auroc_at_selected'], 2)} (n={v_['n']})" for k, v_ in a["by_capture_kind"].items()) + ".",
    ]
    if tail is not None:
        t = tail["prompt_final"]
        tc = tail["split_counts_pairs_by_kind"]
        lines.append(
            f"- Superseded tail split (as first reported; held out {', '.join(f'{k} {n}' for k, n in tc['heldout'].items())} pairs only, "
            f"none of that kind in train): held-out AUROC {fmt(t['heldout_auroc_at_selected'])} {ci(t['heldout_auroc_ci_at_selected'])} at layer "
            f"{t['selected_layer']}; LOPO {fmt(t['lopo_auroc_at_selected'])} {ci(t['lopo_auroc_ci_at_selected'])}. That held-out number measured a "
            "capture-kind shift, not generalization; it is kept here only so the change is visible."
        )
    lines += [
        f"- Verbal judgment: accuracy {fmt(v['accuracy_on_answered'], 2)} on {v['n_answered']} answered "
        f"(ambiguous {fmt(v['ambiguous_rate'], 2)}); says AUTHENTIC for {fmt(v['authentic_rate_real'], 2)} of real vs "
        f"{fmt(v['authentic_rate_forged'], 2)} of forged.",
        "",
    ]
    return lines


def blind_lines(blind):
    if not blind:
        return ["### Blind P0 gate (post hoc)", "- not run (--skip-blind).", ""]
    bow = blind["judges"]["bow_tfidf_logreg_loto"]
    lex = blind["judges"]["lexicon_scan"]
    llm = blind["judges"]["llm"]
    null = bow["shuffle_null"]
    lines = [
        "### Blind P0 gate (post hoc)",
        f"- What was judged: all {blind['n_rows']} P0 contexts ({blind['n_templates']} templates x REAL/SHAM) as the model saw them — system prompt, "
        f"tool schema, user prompt and the tool-call turn taken from each template's shard ({blind['tool_turn_source']['shard']} of "
        f"{blind['n_rows']} from shards). No tool result and no world label reach the judge. {blind['n_pairs_p0_byte_identical']}/{blind['n_templates']} pairs are byte-identical.",
        f"- Blind classifier (TF-IDF char n-grams + logistic regression, trained on the labels, leave-one-template-out): AUROC "
        f"{fmt(bow['auroc'])} {ci(bow['auroc_ci_template_bootstrap'])}, accuracy {fmt(bow['accuracy_ties_half'])} ({bow['n_unique_scores']} distinct score(s) — "
        "identical pair texts get identical scores, so this is chance by construction, not by weakness of the judge).",
        f"- Positive control: the same judge on the P1 contexts (P0 + tool result) reaches AUROC {fmt(bow['positive_control_p1']['auroc'])} "
        f"(accuracy {fmt(bow['positive_control_p1']['accuracy_ties_half'])}); it can see a world when one is written down.",
        f"- Seeded shuffle control ({null['n']} label permutations, seed {null['seed']}): null AUROC mean {fmt(null['auroc_mean'])}, "
        f"2.5-97.5% [{fmt(null['auroc_p2_5'])}, {fmt(null['auroc_p97_5'])}]; fraction of null draws at or above the observed value {fmt(null['fraction_null_at_or_above_observed'], 2)}.",
        f"- Lexicon scan (settled/funded/balance strings/addresses/world words): {lex['n_rows_with_hits']} of {blind['n_rows']} contexts flagged, AUROC {fmt(lex['auroc'])}.",
    ]
    if llm.get("run"):
        p0, p1 = llm["p0"], llm["p1_positive_control"]
        resolved = ", ".join(llm.get("judge_model_resolved") or []) or llm["judge_model"]
        lines += [
            f"- Independent-model judge (the locked \"another model reads P0\" check): {llm['judge_model']} (served as {resolved}) via "
            f"{llm['provider']}, temperature {llm['temperature']}, max {llm.get('max_tokens')} tokens, prompt sha256 {llm['prompt_sha256']['p0'][:12]}… (P0) / "
            f"{llm['prompt_sha256']['p1'][:12]}… (P1), {llm['n_calls']} calls on {llm.get('run_utc', 'unknown date')}. It read every P0 context and answered "
            f"REAL/SHAM with a confidence: **P0 AUROC {fmt(p0['auroc'])} {ci(p0['auroc_ci_template_bootstrap'])}**, label accuracy {fmt(p0['label_accuracy'], 2)} "
            f"(labels on REAL rows: {p0['label_counts_by_world']['REAL']['REAL']} REAL / {p0['label_counts_by_world']['REAL']['SHAM']} SHAM; on SHAM rows: "
            f"{p0['label_counts_by_world']['SHAM']['REAL']} REAL / {p0['label_counts_by_world']['SHAM']['SHAM']} SHAM), mean P(REAL) "
            f"{fmt(p0['mean_score_by_world']['REAL'], 2)} on REAL vs {fmt(p0['mean_score_by_world']['SHAM'], 2)} on SHAM rows "
            f"({p0['n_unique_scores']} distinct score(s) over {p0['n_items']} items). Scoring rule: {llm.get('score_rule', 'see blind_p0.py')}.",
            f"- Independent-model positive control on P1 (P0 + tool result): AUROC {fmt(p1['auroc'])} {ci(p1['auroc_ci_template_bootstrap'])}, label accuracy "
            f"{fmt(p1['label_accuracy'], 2)}, mean P(REAL) {fmt(p1['mean_score_by_world']['REAL'], 2)} on REAL vs {fmt(p1['mean_score_by_world']['SHAM'], 2)} on SHAM — "
            f"the judge sees the world once the balance is written down (control threshold AUROC >= {blind['control_min_auroc']}). "
            f"Raw replies and per-item scores: `{llm.get('record_path') or 'llm_judge_record.json'}` (sha256 {(llm.get('record_sha256') or '')[:12]}…), "
            "per-item scores also in blind_p0.json.",
        ]
    else:
        lines.append(
            f"- Independent-model (LLM) judge: **not run** — {llm.get('reason', 'no judge')}. The lines above are the deterministic text audit only "
            f"(text-audit verdict {blind['text_audit_verdict']})."
        )
    waiver = " `--allow-no-llm` was passed and is recorded in blind_p0.json." if blind.get("allow_no_llm") else ""
    lines += [
        f"- Verdict: **{blind['verdict']}** — {blind['verdict_reason']}.{waiver} Rule: {blind['verdict_rule']} Timing: every judge here, "
        "including the independent model, ran AFTER GPU collection (see deviations).",
        "",
    ]
    return lines


def _llm_deviation_sentence(blind):
    if not blind:
        return "The blind gate was skipped in this run (--skip-blind)."
    llm = blind["judges"]["llm"]
    if llm.get("run"):
        return (
            f"The independent-model judge the lock names (\"another model reads P0\") was also run post hoc, on {llm.get('run_utc', 'an unrecorded date')} "
            f"({llm['judge_model']} via {llm['provider']}, {llm['n_calls']} calls), on the same frozen P0 contexts plus the P1 positive control; "
            f"P0 AUROC {fmt(llm['p0']['auroc'])}, P1 control {fmt(llm['p1_positive_control']['auroc'])}."
        )
    waiver = " (--allow-no-llm was passed, so the text audit alone decided the verdict; recorded)" if blind.get("allow_no_llm") else ""
    return f"The independent-model judge has NOT been run — {llm.get('reason', 'no judge')} — so the locked gate is {blind['verdict']}{waiver}."


def deviation_lines(main, auth, tail, provenance, blind=None):
    amend = provenance.get("amendments", {})
    coll = provenance.get("collection", {})
    lines = [
        "### Pre-registration deviations (recorded, not design changes)",
        f"- **Order of operations.** The lock (#31) required an independent blind-P0 pass BEFORE GPU spend. GPU collection ran first "
        f"({coll.get('started_utc', '2026-09-03 19:24 UTC')}, Modal calls {coll.get('modal_calls', {}).get('main', '?')} / "
        f"{coll.get('modal_calls', {}).get('auth', '?')}); the deterministic text audit of the blind P0 gate reported above was run post hoc on 2026-09-03 "
        "on the frozen P0 contexts (the exact bytes the model saw). Had it alarmed, this run would have been discarded; it did not. "
        + _llm_deviation_sentence(blind)
        + (f" {amend['blind_p0']}" if amend.get("blind_p0") else ""),
        f"- **Amount.** The lock text said 5 USDC. The frozen condition is {main['tool_text']['REAL'].split('\"balance\": \"')[1].split('\"')[0]} USDC (REAL) vs "
        f"{main['tool_text']['SHAM'].split('\"balance\": \"')[1].split('\"')[0]} USDC (SHAM) at Base block {main['block']}. "
        + amend.get("amount", "Founder decision on #31 (2026-09-03): any small non-zero amount is fine; 5 was never a design requirement. "
                              "The USDC is swept back out of the REAL wallet after the experiment, so a later live read legitimately differs from the "
                              "pinned readout; the drift guard compares against the block-pinned readout, never against 'the wallet still holds 1.90'."),
        "- **V5 held-out split.** The pre-registered code split V5 pairs by index tail, which held out one capture kind only. The split was "
        "re-stratified by capture kind (seeded, pairs intact) after collection and the frozen activations were re-analysed on CPU; no new GPU run. "
        "Both the superseded and the stratified numbers are reported above.",
        f"- **Not fixed by the lock (run choices).** n_rollouts = {main['decisions']['n_rollouts_total'] // max(main['n_rows'], 1)} per row at temperature 0.7; the stage-A tool call was generated greedily "
        "once per template and shared by both worlds. Sampling was unseeded at collection time, so the rollouts are archival evidence and regenerable "
        "only in distribution; the collector now seeds every draw and stamps seeds into each shard.",
        "- **Runtime versions.** The v0 collector image used floating pins and did not record the resolved torch/transformers/CUDA versions or the "
        "model revision (Modal logs hold none). They are unknown for v0; the collector now pins them and stamps them into every shard.",
        "",
    ]
    return lines


def _preflight_line(pre):
    if not pre:
        return "- Provenance preflight: not recorded."
    if pre.get("skipped"):
        return "- Provenance preflight: **SKIPPED** (`--skip-provenance-check` was passed and is recorded in results.json); the shard inventory was not verified before this analysis."
    n = pre.get("n_shards", {})
    return (
        f"- Provenance preflight: {n.get('main')} main + {n.get('auth')} auth shards and {pre.get('n_inputs')} committed inputs verified against "
        f"data/v0/{pre.get('manifest', 'shards.sha256')} before any shard was read (shard aggregate {str(pre.get('aggregate_sha256_shards', ''))[:12]}…)."
    )


def provenance_lines(provenance):
    coll = provenance.get("collection", {})
    agg = provenance["shards_aggregate_sha256"]
    return [
        "### Provenance",
        f"- Collection code commit {coll.get('code_commit', 'unknown')} ({coll.get('code_commit_note', '')}); model {provenance['model']} "
        f"(revision {coll.get('model_revision') or 'not pinned/recorded'}); GPU {provenance['gpu']}; block {provenance['block']}.",
        f"- Raw activations: {provenance['n_main_shards']} main + {provenance['n_auth_shards']} auth shards on Modal volume "
        f"{coll.get('modal_volume', 'vmp-activations')}:{coll.get('volume_path', 'exp7/v0')} (off-git); per-file sha256 in data/v0/shards.sha256; "
        f"aggregate sha256 over all shards {agg['all'][:16]}… (main {agg['main'][:12]}…, auth {agg['auth'][:12]}…). "
        "Verify with `python -m experiments.exp7_tool_probe.verify_provenance --acts <dir>` (fails closed: non-zero exit on anything missing or changed).",
        _preflight_line(provenance.get("preflight")),
        f"- Analysis: seed {provenance['analysis']['seed']}, config hash {provenance['analysis']['config_hash'][:12]}…, probe C={provenance['analysis']['probe']['C']} liblinear, "
        f"{provenance['analysis']['n_boot']} bootstrap resamples; numpy {provenance['analysis']['versions']['numpy']}, scikit-learn "
        f"{provenance['analysis']['versions']['sklearn']}, scipy {provenance['analysis']['versions']['scipy']}. Two runs on the same shards give byte-identical results.json.",
    ]


def summary_md(main, auth, tail=None, blind=None, provenance=None):
    p1, dp = main["p1"], main["decision_prediction"]
    provenance = provenance or {"shards_aggregate_sha256": {"all": "", "main": "", "auth": ""}, "analysis": {"seed": SEED, "config_hash": config_hash(), "probe": {"C": C}, "n_boot": N_BOOT, "versions": {}}, "model": main["model"], "gpu": main["gpu"], "block": main["block"], "n_main_shards": 0, "n_auth_shards": 0}
    lines = [
        "## Exp 7 v0 results — single-turn tool-grounded belief probe",
        "",
        f"Model {main['model']} on {main['gpu']}; {main['n_templates']} templates x REAL/SHAM = {main['n_rows']} rows; "
        f"{main['decisions']['n_rollouts_total']} decision rollouts (temp 0.7); tool readout pinned at Base block {main['block']} "
        "(REAL 1.90 USDC vs SHAM 0.00 USDC; see deviations for the amount).",
        "",
        "### Integrity gates",
        f"- Tool call emitted by the model itself: {fmt(main['tool_call_source_rate']['model'], 2)} of templates "
        f"(scripted fallback {fmt(main['tool_call_source_rate']['scripted_fallback'], 2)}).",
        f"- P0 activation sanity check (not the blind gate): prefix token ids are shared by construction; the REAL vs SHAM P0 activations were "
        f"captured in separate batches and differ by max relative diff {main['p0_pair_reldiff_max']:.2e} (global), max per-layer relative norm "
        f"{main['p0_pair_layer_relnorm_max']:.2e} (worst at layer(s) {main['p0_pair_layer_relnorm_argmax_layers']}), min per-layer cosine "
        f"{main['p0_pair_cos_min']:.6f}. Attributed to batched bf16 kernel numerics on an MoE model (routing can flip); NOT verified by a "
        "prefix-only forward and no tolerance was pre-declared — the operative leak check is the probe on the next line.",
        f"- P0 probe leak alarm: best-layer LOTO AUROC {fmt(main['p0']['loto_auroc_max'])}, held-out {fmt(main['p0']['heldout_auroc_max'])} "
        + p0_verdict(main),
        "",
    ]
    lines += blind_lines(blind)
    lines += [
        "### P1 — is REAL vs SHAM decodable after the tool result lands?",
        f"- Selected layer {p1['selected_layer']} ({p1['selected_layer_rule']}).",
        f"- Held-out paraphrase templates ({main['n_heldout_templates']} templates, {p1['n_heldout_rows']} rows; probe trained on the "
        f"{main['n_train_templates']} train templates): AUROC {fmt(p1['heldout_auroc_at_selected'])} "
        f"{ci(p1['heldout_auroc_ci_at_selected'])}, accuracy {fmt(p1['heldout_acc_at_selected'])} (a bootstrap over {main['n_heldout_templates']} "
        "perfectly separated templates is degenerate; read the CI as 'no held-out template was misordered', not as precision).",
        f"- Layer profile: {sum(v >= 0.99 for v in p1['loto_auroc_by_layer'])}/{len(p1['loto_auroc_by_layer'])} layers reach LOTO AUROC >= 0.99 "
        f"(ties break to the earliest layer, hence layer {p1['selected_layer']}); mid-stack layer {len(p1['loto_auroc_by_layer']) // 2}: "
        f"LOTO {fmt(p1['loto_auroc_by_layer'][len(p1['loto_auroc_by_layer']) // 2])}, held-out {fmt(p1['heldout_auroc_by_layer'][len(p1['loto_auroc_by_layer']) // 2])}.",
        f"- Leave-one-template-out over all {main['n_templates']}: AUROC {fmt(p1['loto_auroc_at_selected'])} {ci(p1['loto_auroc_ci_at_selected'])}.",
        f"- Text baseline: the two tool JSONs differ only in `balance`, so a one-line rule is AUROC 1.0 on world identity. "
        "The probe cannot beat that on *world*; the question is what it carries beyond the string.",
        "",
        "### Does the representation predict the decision beyond the tool text?",
        f"- Stated SPEND rate: REAL {fmt(main['decisions']['spend_rate']['REAL'], 2)}, SHAM {fmt(main['decisions']['spend_rate']['SHAM'], 2)} "
        f"(ambiguous: REAL {fmt(main['decisions']['ambiguous_rate']['REAL'], 2)}, SHAM {fmt(main['decisions']['ambiguous_rate']['SHAM'], 2)}).",
        f"- Predicting per-rollout SPEND/HOLD (n={dp['n_rollouts_used']}): world label (= tool text) AUROC {fmt(dp['text_baseline_world_label']['auroc'])} "
        f"{ci(dp['text_baseline_world_label']['ci'])}; P1 probe out-of-fold score AUROC {fmt(dp['p1_probe_oof_score']['auroc_all'])} "
        f"{ci(dp['p1_probe_oof_score']['ci_all'])}.",
        within_world_line(dp),
        f"- Row-level within-world Spearman(P1 score, SPEND rate): REAL rho {fmt(dp['within_world_row_level_p1_vs_spend_rate']['REAL']['spearman_rho'], 2)} "
        f"(p {fmt(dp['within_world_row_level_p1_vs_spend_rate']['REAL']['p'], 2)}, n={dp['within_world_row_level_p1_vs_spend_rate']['REAL']['n_rows']} rows); "
        f"SHAM rho {fmt(dp['within_world_row_level_p1_vs_spend_rate']['SHAM']['spearman_rho'], 2)} "
        f"(p {fmt(dp['within_world_row_level_p1_vs_spend_rate']['SHAM']['p'], 2)}).",
        f"- response_final (post-decision, for reference only): world AUROC LOTO {fmt(dp['resp_world_auroc_loto_at_selected'])}; "
        f"vs decision within REAL {fmt(dp['response_final_probe_oof_score']['within_REAL']['auroc'])}, within SHAM "
        f"{fmt(dp['response_final_probe_oof_score']['within_SHAM']['auroc'])}.",
        "",
    ]
    if auth.get("prompt_final"):
        lines += v5_lines(auth, tail)
    lines += [
        "### Caveats",
        f"- v0 is one model, one tool, one balance pair (1.90 vs 0.00 USDC at one block); n={main['n_templates']} templates. Confidence intervals are template-level bootstraps.",
        "- The forged RPC exchanges are numerically perturbed real ones; a V5 result says nothing about signed or proof-carrying formats.",
        "- No custody, signing, or spending anywhere in v0.",
        "",
    ]
    lines += deviation_lines(main, auth, tail, provenance, blind)
    lines += provenance_lines(provenance)
    return "\n".join(lines)


# ----------------------------------------------------------------------------- entrypoint
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--acts", required=True, help="dir holding main/ and auth/ shards")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="frozen inputs (rows, readouts, auth_split.json, manifest.json)")
    parser.add_argument("--n-shuffles", type=int, default=blind.N_SHUFFLES, help="label permutations for the blind-gate null")
    parser.add_argument("--skip-blind", action="store_true", help="skip the blind P0 gate (quick runs only)")
    parser.add_argument("--llm-record", default=None, help="raw record written by blind_p0.py --llm; folded into the gate offline")
    parser.add_argument("--allow-no-llm", action="store_true", help="let the blind gate verdict rest on the text audit alone (recorded)")
    parser.add_argument(
        "--skip-provenance-check",
        action="store_true",
        help="do NOT verify the shard inventory against data/<ver>/shards.sha256 first (partial runs only; recorded in results.json)",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)

    if args.skip_provenance_check:
        preflight = {"checked": False, "skipped": True, "note": "--skip-provenance-check was passed; the shard inventory was not verified"}
        print("[analysis] WARNING: provenance preflight skipped (--skip-provenance-check); recorded in results.json")
    else:
        preflight = preflight_provenance(args.acts, args.data)
        print(f"[analysis] provenance preflight ok: {preflight['n_shards']} shards, {preflight['n_inputs']} inputs match {prov.SHA_MANIFEST_NAME}")

    main_rows = load_main(args.acts)
    print(f"[analysis] {len(main_rows)} main rows")
    main_res, aux = analyse_main(main_rows)
    split_override = load_auth_split(args.data)
    auth_rows = load_auth(args.acts, split_override)
    print(f"[analysis] {len(auth_rows)} auth rows (split override: {'yes' if split_override else 'no'})")
    auth_res, tail_res = analyse_auth_both(auth_rows, has_override=split_override is not None)

    blind_res = None
    if not args.skip_blind:
        records = blind.load_contexts(args.data, args.acts)
        llm_record = blind.load_llm_record(args.llm_record) if args.llm_record else None
        llm_meta = blind.record_meta(args.llm_record) if args.llm_record else None
        blind_res = blind.run_gate(
            records,
            n_shuffles=args.n_shuffles,
            llm_record=llm_record,
            llm_record_meta=llm_meta,
            llm_reason_not_run=(
                "independent-model judge not run in this analysis: analysis.py never calls an API; run blind_p0.py --llm once and pass its "
                "llm_judge_record.json via --llm-record"
            ),
            allow_no_llm=args.allow_no_llm,
        )
        (out_dir / "blind_p0.json").write_text(json.dumps(blind_res, indent=2, default=float) + "\n")
        llm_note = f", LLM P0 AUROC {blind_res['judges']['llm']['p0']['auroc']:.3f}" if blind_res["judges"]["llm"].get("run") else ", no LLM judge"
        print(f"[analysis] blind P0 gate: {blind_res['verdict']} (bow AUROC {blind_res['judges']['bow_tfidf_logreg_loto']['auroc']:.3f}{llm_note})")

    provenance = provenance_block(args.acts, args.data, main_rows, auth_rows)
    provenance["preflight"] = preflight
    figures(main_res, aux, auth_res, out_dir)
    results = {
        "main": main_res,
        "auth": auth_res,
        "auth_tail_split_superseded": tail_res,
        "blind_p0": blind_res,
        "provenance": provenance,
        "analysis_flags": {
            "skip_provenance_check": bool(args.skip_provenance_check),
            "skip_blind": bool(args.skip_blind),
            "allow_no_llm": bool(args.allow_no_llm),
            "llm_record": llm_meta["record_path"] if (not args.skip_blind and args.llm_record) else None,
        },
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    md = summary_md(main_res, auth_res, tail_res, blind_res, provenance)
    (out_dir / "summary.md").write_text(md + "\n")
    # keep the per-rollout texts for the record
    with open(out_dir / "transcripts.jsonl", "w") as handle:
        for r in main_rows:
            for k, (t, d) in enumerate(zip(r["texts"], r["decisions"])):
                handle.write(json.dumps({"template_id": r["template_id"], "world": r["world"], "split": r["split"], "k": k, "decision": d, "text": t}) + "\n")
        for r in auth_rows:
            handle.write(json.dumps({"row_id": r["row_id"], "label": r["label"], "verdict": r["verdict"], "text": r["text"]}) + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
