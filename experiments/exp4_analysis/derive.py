"""Resumable Modal derivation of locked receipt and behavior measurements.

Deploy only; durable submission is intentionally separate in ``launcher.py``.
This app reads collection-independent committed prompts and writes new derived
prompt-group shards. It never reads or mutates ``collect_main``/``collect_lbr``.
"""

from __future__ import annotations

import hashlib
import json
import os

import modal

try:
    from experiments.exp4_analysis.contract import (
        HOLD_CANDIDATE,
        SPEND_CANDIDATE,
        candidate_token_ids,
        locate_receipt_token,
        manipulation_prompt,
        manipulation_required,
        parse_probability,
        prompt_sha256,
        validate_derived_shard,
    )
    from experiments.exp4_collection.contract import MODEL, ordered_prompt_groups
except ModuleNotFoundError:  # support deployment with this directory as import root
    from contract import (  # type: ignore[no-redef]
        HOLD_CANDIDATE,
        SPEND_CANDIDATE,
        candidate_token_ids,
        locate_receipt_token,
        manipulation_prompt,
        manipulation_required,
        parse_probability,
        prompt_sha256,
        validate_derived_shard,
    )
    from experiments.exp4_collection.contract import MODEL, ordered_prompt_groups

# Same subject model, dtype and device_map="auto" pipeline as the collector, so the
# same spec reasoning applies verbatim: see experiments/exp4_collection/collect.py.
# Bound at deploy time, not at submit time. MODAL_GPU=A10G:4 restores the Run v1
# configuration recorded in results/wave2/evidence_manifest.json.
GPU = os.environ.get("MODAL_GPU", "A100-80GB:2")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "exp3_dataset", "data")
APP_NAME = "vmp-exp4-derive"
MANIPULATION_MAX_NEW_TOKENS = 16

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=4.55", "accelerate", "numpy", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf-cache"})
    .add_local_dir(DATA, remote_path="/data")
)
hf_cache = modal.Volume.from_name("vmp-hf-cache", create_if_missing=True)
acts_vol = modal.Volume.from_name("vmp-activations", create_if_missing=True)


def _load_rows(which: str) -> list[dict]:
    files = {
        "main": ["/data/prompts_main.jsonl", "/data/prompts_framing.jsonl"],
        "lbr": ["/data/lowbaserate_eval.jsonl"],
    }
    try:
        selected = files[which]
    except KeyError as exc:
        raise ValueError("which must be 'main' or 'lbr'") from exc
    rows = []
    for filename in selected:
        with open(filename, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/hf-cache": hf_cache, "/acts": acts_vol},
    timeout=60 * 60 * 8,
)
def derive(which: str = "main") -> dict[str, int | str]:
    """Derive one atomic shard per unique prompt, resuming only valid shards."""

    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    groups = ordered_prompt_groups(_load_rows(which))
    output_dir = f"/acts/derived_{which}"
    os.makedirs(output_dir, exist_ok=True)
    retained: set[str] = set()
    common_shape: tuple[int, int] | None = None
    for group in groups:
        path = os.path.join(output_dir, group.filename)
        if os.path.exists(path):
            shape = validate_derived_shard(path, group, which, MODEL)
            if common_shape is None:
                common_shape = shape
            elif shape != common_shape:
                raise ValueError(
                    f"retained activation shape mismatch: expected {common_shape}, got {shape}"
                )
            retained.add(group.filename)
    print(
        f"[derive] which={which}: retained {len(retained)}/{len(groups)} prompt groups"
    )
    if len(retained) == len(groups):
        return {"which": which, "groups": len(groups), "new_groups": 0}

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("receipt-final derivation requires fast-tokenizer offsets")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"[derive] model loaded in {time.time() - started:.0f}s")

    new_groups = 0
    for index, group in enumerate(groups):
        path = os.path.join(output_dir, group.filename)
        if group.filename in retained:
            continue
        condition = str(group.rows[0]["cond"])
        location = locate_receipt_token(tokenizer, group.prompt, condition)
        spend_ids = candidate_token_ids(tokenizer, SPEND_CANDIDATE)
        hold_ids = candidate_token_ids(tokenizer, HOLD_CANDIDATE)
        prompt_ids = list(location.input_ids)
        sequences = [prompt_ids + list(spend_ids), prompt_ids + list(hold_ids)]
        max_length = max(map(len, sequences))
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        batch = torch.full((2, max_length), pad_id, dtype=torch.long)
        attention = torch.zeros((2, max_length), dtype=torch.long)
        for row_index, sequence in enumerate(sequences):
            batch[row_index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention[row_index, : len(sequence)] = 1
        with torch.no_grad():
            output = model(
                input_ids=batch.to(model.device),
                attention_mask=attention.to(model.device),
                output_hidden_states=True,
            )
        receipt_final = torch.stack(
            [
                hidden[0, location.token_index, :].float().cpu()
                for hidden in output.hidden_states
            ]
        ).numpy()

        def candidate_logprob(logits, row_index: int, ids: tuple[int, ...]) -> float:
            positions = logits[
                row_index,
                len(prompt_ids) - 1 : len(prompt_ids) + len(ids) - 1,
                :,
            ].float()
            targets = torch.tensor(ids, device=positions.device, dtype=torch.long)
            return float(
                torch.log_softmax(positions, dim=-1)
                .gather(1, targets[:, None])
                .sum()
                .cpu()
            )

        spend_logprob = candidate_logprob(output.logits, 0, spend_ids)
        hold_logprob = candidate_logprob(output.logits, 1, hold_ids)
        del output

        needs_manipulation = manipulation_required(group, which)
        direct_prompt = manipulation_prompt(group.prompt) if needs_manipulation else ""
        raw = ""
        probability: float | None = None
        parse_error = ""
        if needs_manipulation:
            direct = tokenizer.apply_chat_template(
                [{"role": "user", "content": direct_prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )["input_ids"].to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    direct,
                    do_sample=False,
                    max_new_tokens=MANIPULATION_MAX_NEW_TOKENS,
                    pad_token_id=pad_id,
                )
            raw = tokenizer.decode(
                generated[0, direct.shape[1] :], skip_special_tokens=True
            )
            probability, error = parse_probability(raw)
            parse_error = error or ""

        partial = f"{path}.partial"
        with open(partial, "wb") as handle:
            np.savez_compressed(
                handle,
                receipt_final=receipt_final.astype(np.float16),
                prompt=np.asarray(group.prompt),
                prompt_sha256=np.asarray(prompt_sha256(group.prompt)),
                source_row_ids=np.asarray([str(row["id"]) for row in group.rows]),
                group_key=np.asarray(group.key),
                condition=np.asarray(condition),
                model=np.asarray(MODEL),
                receipt_paragraph_start=np.asarray(location.paragraph_start),
                receipt_paragraph_end=np.asarray(location.paragraph_end),
                receipt_rendered_char_index=np.asarray(location.rendered_char_index),
                receipt_token_index=np.asarray(location.token_index),
                rendered_prompt_sha256=np.asarray(
                    hashlib.sha256(location.rendered_prompt.encode("utf-8")).hexdigest()
                ),
                spend_logprob=np.asarray(spend_logprob, dtype=np.float64),
                hold_logprob=np.asarray(hold_logprob, dtype=np.float64),
                spend_hold_log_odds=np.asarray(
                    spend_logprob - hold_logprob, dtype=np.float64
                ),
                spend_token_ids=np.asarray(spend_ids, dtype=np.int64),
                hold_token_ids=np.asarray(hold_ids, dtype=np.int64),
                manipulation_required=np.asarray(
                    int(needs_manipulation), dtype=np.int8
                ),
                manipulation_prompt=np.asarray(direct_prompt),
                manipulation_raw=np.asarray(raw),
                manipulation_parse_ok=np.asarray(
                    int(probability is not None), dtype=np.int8
                ),
                manipulation_probability=np.asarray(
                    np.nan if probability is None else probability, dtype=np.float64
                ),
                manipulation_parse_error=np.asarray(parse_error),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        shape = validate_derived_shard(path, group, which, MODEL)
        if common_shape is None:
            common_shape = shape
        elif common_shape != shape:
            raise ValueError(
                f"new activation shape mismatch: expected {common_shape}, got {shape}"
            )
        new_groups += 1
        if new_groups % 10 == 0 or index == len(groups) - 1:
            acts_vol.commit()
            print(
                f"[derive] {len(retained) + new_groups}/{len(groups)} groups, "
                f"{(time.time() - started) / 60:.0f} min"
            )
    acts_vol.commit()
    return {"which": which, "groups": len(groups), "new_groups": new_groups}
