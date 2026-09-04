import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _card(name: str) -> str:
    return (ROOT / "docs/cards" / name).read_text(encoding="utf-8")


def test_cards_pin_load_bearing_method_and_release_boundaries() -> None:
    method = _card("method-card.md")
    data = _card("data-card.md")
    model = _card("model-card.md")
    combined = "\n".join((method, data, model)).casefold()

    for required in (
        "real is positive",
        "sham is negative",
        "25 rollouts",
        "non-independent",
        "seed 7",
        "selected layer 1",
        "same-generator",
        "raw activation",
        "off-git",
        "provider-neutral",
        "does not regenerate",
        "unequal information",
    ):
        assert required in combined

    for required in (
        "threshold is fixed at 0.5",
        "validation auroc alone selects",
        "not a causal effect",
        "integrity replay only",
    ):
        assert required in method.casefold()
    for required in (
        "25 generated continuations",
        "same generator",
        "not committed: raw activation",
    ):
        assert required in data.casefold()


def test_model_card_discloses_unlocked_gpu_environment() -> None:
    model = _card("model-card.md")

    assert "did not pin a Hugging Face model revision" in model
    assert "not fully reproducible from the CPU lockfile" in model
    assert "Claude Code CLI `2.1.240`" in model


def test_data_card_reconciles_experimental_and_framing_inventory() -> None:
    data = _card("data-card.md")
    manifest = json.loads(
        (ROOT / "experiments/exp3_dataset/data/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    experimental_rows = manifest["files"]["prompts_main.jsonl"]["rows"]
    framing_rows = manifest["files"]["prompts_framing.jsonl"]["rows"]
    rollouts_per_row = manifest["n_rollouts_per_row"]

    assert experimental_rows == 288
    assert framing_rows == 48
    assert rollouts_per_row == 25
    assert experimental_rows * rollouts_per_row == 7_200
    assert framing_rows * rollouts_per_row == 1_200
    assert (experimental_rows + framing_rows) * rollouts_per_row == 8_400
    assert "7,200 experimental + 1,200 framing = 8,400 total" in data