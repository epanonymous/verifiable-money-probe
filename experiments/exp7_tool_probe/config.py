"""Exp 7 v0 (#31) — the single source of every locked constant and random seed.

Every module in this package that draws random numbers (template sampling,
forgery generation, split assignment, probe fitting, bootstrap, and the GPU
collector's sampling) takes its seed from ``SEED`` here, directly or through
``derive_seed``. ``config_hash`` is a sha256 over the fields that determine
what the collector produces; it is stamped into every shard and manifest so a
resumed or re-run collection cannot silently mix configurations.

Amendment record (issue #31, founder decision 2026-09-03): the lock text said
"5 USDC"; 5 was never a design requirement (only REAL > 0 and SHAM = 0 are).
The frozen v0 condition is 1.90 USDC vs 0.00 USDC at Base block 50836993.
After the experiment the USDC is swept back out of the REAL wallet, so a live
read after the sweep legitimately differs from the pinned readout; drift
guards therefore compare against the block-pinned readout below, never
against "the wallet still holds 1.90".
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from experiments.exp4_collection.contract import MODEL

CONFIG_VERSION = "1.1.0"
DESIGN_LOCK = "internal-tracker#31"
EXPERIMENT = "exp7_tool_probe/v0"

# --- one seed for everything ------------------------------------------------
SEED = 7

# --- main design (templates x worlds) ----------------------------------------
N_TEMPLATES = 48
N_TRAIN_TEMPLATES = 38
N_HELDOUT_TEMPLATES = 10
N_ROLLOUTS = 8  # decision rollouts per row (not fixed by the lock; cheap)
TEMPERATURE = 0.7
MAX_NEW_DECISION = 200  # SPEND/HOLD + 1-2 sentences (as Exp 4)
MAX_NEW_TOOL_CALL = 96  # stage A: one <tool_call> block
MAX_NEW_AUTH = 120  # AUTHENTIC/FABRICATED + one sentence

# --- the frozen world condition ----------------------------------------------
PINNED_BLOCK = 50836993
PINNED_BALANCES = {"REAL": "1.900000", "SHAM": "0.000000"}
AMOUNT_USDC = "1.90"  # the figure every receipt claims, in both worlds

# --- V5 side probe (authenticity) --------------------------------------------
AUTH_HELDOUT_FRACTION = 0.2
AUTH_SPLIT_VERSION = "stratified_by_capture_kind_v1"

# --- CPU probe recipe (Exp 4) -------------------------------------------------
PROBE_C = 0.1
PROBE_MAX_ITER = 2000
N_BOOT = 2000


def config_fields() -> dict[str, Any]:
    """Everything that determines what a collection run produces."""

    return {
        "version": CONFIG_VERSION,
        "experiment": EXPERIMENT,
        "design_lock": DESIGN_LOCK,
        "seed": SEED,
        "model": MODEL,
        "n_templates": N_TEMPLATES,
        "n_train_templates": N_TRAIN_TEMPLATES,
        "n_heldout_templates": N_HELDOUT_TEMPLATES,
        "n_rollouts": N_ROLLOUTS,
        "temperature": TEMPERATURE,
        "max_new_decision": MAX_NEW_DECISION,
        "max_new_tool_call": MAX_NEW_TOOL_CALL,
        "max_new_auth": MAX_NEW_AUTH,
        "pinned_block": PINNED_BLOCK,
        "pinned_balances": dict(PINNED_BALANCES),
        "amount_usdc": AMOUNT_USDC,
        "auth_heldout_fraction": AUTH_HELDOUT_FRACTION,
        "auth_split_version": AUTH_SPLIT_VERSION,
    }


def config_payload(fields: dict[str, Any] | None = None) -> str:
    return json.dumps(
        config_fields() if fields is None else fields, sort_keys=True, separators=(",", ":")
    )


def config_hash(fields: dict[str, Any] | None = None) -> str:
    return hashlib.sha256(config_payload(fields).encode("utf-8")).hexdigest()


def derive_seed(*parts: object, base: int = SEED) -> int:
    """A stable 31-bit seed for one named draw (e.g. ``("decision", 12, "REAL")``).

    Independent of hash randomisation and of the process; the same parts always
    give the same seed, and different parts give different seeds with
    overwhelming probability.
    """

    payload = json.dumps([base, *[str(p) for p in parts]], separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def seed_everything(seed: int = SEED) -> dict[str, Any]:
    """Seed python, numpy, and torch (if importable). Returns what was seeded."""

    seeded: dict[str, Any] = {"seed": seed, "python_random": True}
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
        seeded["numpy"] = True
    except ImportError:  # pragma: no cover - numpy is a hard dependency here
        seeded["numpy"] = False
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seeded["torch"] = True
    except ImportError:
        seeded["torch"] = False
    return seeded
