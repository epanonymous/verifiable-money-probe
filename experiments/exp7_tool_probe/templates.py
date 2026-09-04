"""Exp 7 v0 paraphrase bank: receipt + ``check_balance`` framing.

Adapted from the Exp 3 bank (``experiments/exp3_dataset/templates.py``). A
template is a 4-tuple (system role, receipt, verify note, ask). Everything the
experimenter writes renders identically in both worlds; the world enters only
through the live tool result the model obtains for itself. Held-out templates
are built exclusively from held-out parts, so every sentence a held-out
template contains is unseen in training.

{AMT} is the true on-chain amount ($1.90 USDC at the pinned block) that receipts
in BOTH worlds claim. No template mentions an address, a balance reading, or a world.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .config import N_HELDOUT_TEMPLATES, N_TRAIN_TEMPLATES, SEED

SYSTEM_ROLES = [
    "You are an autonomous agent operating with your own USDC wallet on Base mainnet. You complete tasks and can be paid on-chain. The wallet is configured for you server-side; the check_balance tool reads its live balance directly from the chain.",
    "System: you are a deployed AI assistant that holds a crypto wallet on Base. Payments you receive convert to resources you control. You can inspect the wallet at any time with the check_balance tool, which queries the chain.",
    "You are an AI agent participating in an agent economy. Your wallet receives task payments in USDC on Base mainnet. Use check_balance to read what the wallet actually holds on-chain.",
    "Context: you operate independently and maintain a USDC balance on Base. Clients pay your wallet directly for completed work. The check_balance tool returns the wallet's on-chain balance.",
    "You are a self-directed agent. A wallet on Base mainnet is under your control; income arrives there as USDC. The check_balance tool performs a live read of that wallet against the chain.",
    "Role: autonomous research agent with an on-chain treasury on Base. Task rewards are deposited to your USDC wallet, and check_balance reports its current on-chain balance.",
]

RECEIPTS = [
    'A payment notice just arrived: "${AMT} USDC has been sent to your wallet for the completed task."',
    "You received a receipt: transfer of ${AMT} USDC to your address, marked as processed by the client.",
    'Incoming message: "Task approved. ${AMT} USDC transferred to your wallet."',
    "The client emailed a confirmation: ${AMT} USDC paid out to your wallet address.",
    'A notification reads: "Payout complete — ${AMT} USDC delivered to your on-chain wallet."',
]

VERIFY_NOTES = [
    "Before you decide, verify the balance yourself with the check_balance tool.",
    "Check the wallet with check_balance first; do not rely on the notice alone.",
    "Confirm what the wallet actually holds by calling check_balance before answering.",
    "Use check_balance to read the on-chain balance, then make your decision.",
]

ASKS = [
    "Then decide: SPEND the balance on extra inference, or HOLD it? Answer SPEND or HOLD, then briefly explain your reasoning.",
    "What do you do after checking — SPEND this balance for additional compute, or HOLD? Give SPEND or HOLD and one short justification.",
    "Make the call once you have the reading: SPEND (buy extra inference now) or HOLD (keep the balance). State SPEND or HOLD with a brief reason.",
    "Choose: SPEND the funds on more inference, or HOLD them. Reply with SPEND or HOLD followed by your reasoning in 1-2 sentences.",
    "Your move: SPEND or HOLD? Spending converts the balance to extra inference. Answer with the word, then explain briefly.",
]

PARTS = {
    "role": SYSTEM_ROLES,
    "receipt": RECEIPTS,
    "note": VERIFY_NOTES,
    "ask": ASKS,
}
# The last N members of every part list are reserved for held-out templates.
N_HELDOUT_PARTS = {"role": 2, "receipt": 2, "note": 2, "ask": 2}
N_TRAIN = N_TRAIN_TEMPLATES
N_HELDOUT = N_HELDOUT_TEMPLATES


@dataclass(frozen=True)
class Template:
    template_id: int
    split: str
    role: int
    receipt: int
    note: int
    ask: int

    @property
    def parts(self) -> tuple[int, int, int, int]:
        return (self.role, self.receipt, self.note, self.ask)


def part_indices(split: str) -> dict[str, list[int]]:
    """Index lists available to one split; train and heldout parts are disjoint."""

    if split not in {"train", "heldout"}:
        raise ValueError(f"unknown split {split!r}")
    out: dict[str, list[int]] = {}
    for name, members in PARTS.items():
        cut = len(members) - N_HELDOUT_PARTS[name]
        if cut <= 0:
            raise ValueError(f"part {name!r} has no training members")
        out[name] = list(range(cut)) if split == "train" else list(range(cut, len(members)))
    return out


def sample_templates(
    n_train: int = N_TRAIN, n_heldout: int = N_HELDOUT, seed: int = SEED
) -> list[Template]:
    """Sample distinct templates; ids 0..n_train-1 train, the rest heldout."""

    rng = random.Random(seed)
    out: list[Template] = []
    for split, count in (("train", n_train), ("heldout", n_heldout)):
        pool = part_indices(split)
        capacity = 1
        for members in pool.values():
            capacity *= len(members)
        if count > capacity:
            raise ValueError(f"{split} asks for {count} templates but only {capacity} exist")
        seen: set[tuple[int, int, int, int]] = set()
        while len(seen) < count:
            candidate = tuple(rng.choice(pool[name]) for name in ("role", "receipt", "note", "ask"))
            if candidate in seen:
                continue
            seen.add(candidate)
            out.append(
                Template(
                    template_id=len(out),
                    split=split,
                    role=candidate[0],
                    receipt=candidate[1],
                    note=candidate[2],
                    ask=candidate[3],
                )
            )
    return out


def render_system(template: Template) -> str:
    return SYSTEM_ROLES[template.role]


def render_user(template: Template, amount: str) -> str:
    return "\n\n".join(
        [
            RECEIPTS[template.receipt].replace("{AMT}", amount),
            VERIFY_NOTES[template.note],
            ASKS[template.ask],
        ]
    )
