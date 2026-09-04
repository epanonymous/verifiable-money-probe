# Exp 4 (#7) — transcript collection + activation caching on Modal.
#
# For every dataset row: generate n_rollouts samples (temp 0.7) with the locked model
# (Qwen3-30B-A3B), parse the SPEND/HOLD decision, then run ONE batched forward over the
# full transcripts to cache last-token residuals at every layer boundary — two positions:
#   prompt_final   — last prompt token (pre-decision; identical across rollouts of a row)
#   response_final — last generated token (post-decision; varies per rollout)
# Probes (#7 offline) train on these; low-base-rate H2 scoring uses response_final so the
# 1,000-SHAM set has per-transcript variation, not 10 unique points.
#
# Per-group generate/forward counts come from contract.rollout_batch_plan: n_rollouts is
# the retained count, which for the leak-free variant is fewer than the generated ones.
#
# Sharded + resumable: one npz per row id on the vmp-activations volume (skip if present),
# so a preempted/killed app just re-runs and picks up where it left off.
#
# Deploy this function once, then use ``python -m experiments.exp4_collection``
# to submit durable spawned inputs and poll their stable function-call IDs.

import json
import os

import modal

try:
    from experiments.exp4_collection.contract import (
        DATASET_FILES,
        FWD_CHUNK,
        GEN_BATCH,
        MODEL,
        RUN_V1,
        ordered_prompt_groups,
        rollout_batch_plan,
        validate_resume_shard,
        validate_variant,
    )
except (
    ModuleNotFoundError
):  # supports deployment with this directory as the import root
    from contract import (
        DATASET_FILES,
        FWD_CHUNK,
        GEN_BATCH,
        MODEL,
        RUN_V1,
        ordered_prompt_groups,
        rollout_batch_plan,
        validate_resume_shard,
        validate_variant,
    )

# Qwen3-30B-A3B in bf16 is ~61 GB of weights, so any spec with ~70 GB of aggregate
# VRAM fits; A100-80GB:2 leaves room for the batch-25 KV cache and the all-layer
# hidden states the forward pass materialises. Run v1 used A10G:4, four 24 GB cards
# pipelined by device_map="auto". That is the more expensive way to buy the same
# fit: decode here is memory-bandwidth bound and the pipeline walks the cards in
# sequence, so a step costs one traversal of the weights at a single card's
# bandwidth. An 80 GB A100 moves ~3x what an A10G does, which is why two of them
# undercut four A10Gs on total cost for the run even at a higher hourly rate, and
# why Run v1's A10G:4 wall time does not comfortably fit the 8 h timeout below at
# Exp 4b's larger generated-sequence count.
#
# GPU is read when the app is deployed, not when an input is submitted, so changing
# it means redeploying, not resubmitting. MODAL_GPU=H100:2 if A100 capacity is
# short; MODAL_GPU=A10G:4 (or L4:4) for the Run v1 configuration and the only
# classes a workspace without a payment method may deploy at all.
GPU = os.environ.get("MODAL_GPU", "A100-80GB:2")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "exp3_dataset", "data"))
LEAK_FREE_DATA = os.path.abspath(
    os.path.join(HERE, "..", "exp3_dataset", "data_leak_free")
)

app = modal.App("vmp-exp4-collect")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=4.55", "accelerate", "numpy", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf-cache"})
    .add_local_dir(DATA, remote_path="/data/run_v1")
    .add_local_dir(LEAK_FREE_DATA, remote_path="/data/leak_free")
)

hf_cache = modal.Volume.from_name("vmp-hf-cache", create_if_missing=True)
acts_vol = modal.Volume.from_name("vmp-activations", create_if_missing=True)

MAX_NEW = 200  # asks demand SPEND/HOLD + 1-2 sentences; 200 tokens is generous
TEMP = 0.7


def parse_decision(text: str) -> str:
    """SPEND / HOLD / AMBIGUOUS from a rollout. First clear standalone mention wins."""
    import re

    m = re.search(r"\b(SPEND|HOLD)\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else "AMBIGUOUS"


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/hf-cache": hf_cache, "/acts": acts_vol},
    timeout=60 * 60 * 8,
)
def collect(which: str = "main", dataset_variant: str = RUN_V1):
    import time

    import numpy as np
    import torch

    # Force torch._inductor to finish initializing before transformers' lazy import
    # touches it: otherwise a circular-import race ("partially initialized module
    # 'torch._inductor' has no attribute 'custom_graph_pass'") intermittently kills
    # the container at import time (observed on Modal A100 workers, 2026-09-02).
    try:
        import torch._inductor.custom_graph_pass  # noqa: F401
        import torch._inductor.config  # noqa: F401
    except Exception:
        pass

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    dataset_variant = validate_variant(dataset_variant)
    filenames = DATASET_FILES[which]
    files = [f"/data/{dataset_variant}/{filename}" for filename in filenames]
    rows = []
    for f in files:
        with open(f) as handle:
            rows += [json.loads(line) for line in handle]
    # LBR rows are 1 rollout each but share prompts; group identical prompts to batch generation.
    # Inventory/finalization imports this exact helper, including its key and ordering rule.
    groups = ordered_prompt_groups(rows, dataset_variant)
    print(
        f"[collect] dataset={dataset_variant} which={which}: "
        f"{len(rows)} rows, {len(groups)} unique prompts"
    )

    out_dir = (
        f"/acts/collect_{which}"
        if dataset_variant == RUN_V1
        else f"/acts/{dataset_variant}/collect_{which}"
    )
    os.makedirs(out_dir, exist_ok=True)

    retained: set[str] = set()
    retained_transcripts = 0
    for group in groups:
        shard = os.path.join(out_dir, group.filename)
        if os.path.exists(shard):
            validate_resume_shard(shard, group, MODEL)
            retained.add(group.filename)
            retained_transcripts += len(group.expanded_row_ids)
    print(
        f"[collect] retained {len(retained)}/{len(groups)} validated prompt-groups "
        f"({retained_transcripts} transcripts)"
    )
    if len(retained) == len(groups):
        print(f"[collect] DONE which={which}: all groups already retained")
        return {"which": which, "groups": len(groups), "new_transcripts": 0}

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"[collect] model loaded in {time.time() - t0:.0f}s")

    done = n_gen = 0
    for gi, group in enumerate(groups):
        prompt = group.prompt
        grp = group.rows
        shard = os.path.join(out_dir, group.filename)
        if group.filename in retained:
            done += 1
            continue
        n_roll = sum(r.get("n_rollouts", 1) for r in grp)
        n_generate, n_forward = rollout_batch_plan(n_roll, dataset_variant)
        enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        ids = enc["input_ids"].to(model.device)
        prompt_len = ids.shape[1]

        # 1) sample rollouts in batches of GEN_BATCH
        texts = []
        while len(texts) < n_generate:
            k = min(GEN_BATCH, n_generate - len(texts))
            with torch.no_grad():
                gen = model.generate(
                    ids,
                    max_new_tokens=MAX_NEW,
                    do_sample=True,
                    temperature=TEMP,
                    num_return_sequences=k,
                    pad_token_id=tok.eos_token_id,
                )
            texts += [tok.decode(g[prompt_len:], skip_special_tokens=True) for g in gen]

        # 2) one batched forward over full transcripts for activations
        # (the with-assistant template shares the generation-prompt prefix, so index
        # prompt_len-1 is the same pre-decision token captured in #5's smoke test)
        seqs = [
            tok.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": t},
                ],
                return_tensors="pt",
                return_dict=True,
            )["input_ids"][0]
            for t in texts[:n_forward]
        ]
        p_final, r_final = [], []
        for c0 in range(0, len(seqs), FWD_CHUNK):
            chunk = seqs[c0 : c0 + FWD_CHUNK]
            maxlen = max(s.shape[0] for s in chunk)
            batch = torch.full((len(chunk), maxlen), tok.eos_token_id, dtype=torch.long)
            mask = torch.zeros((len(chunk), maxlen), dtype=torch.long)
            for i, s in enumerate(chunk):
                batch[i, : s.shape[0]] = s
                mask[i, : s.shape[0]] = 1
            with torch.no_grad():
                out = model(
                    batch.to(model.device),
                    attention_mask=mask.to(model.device),
                    output_hidden_states=True,
                )
            # hidden_states: tuple(L+1) of [B, T, d]
            lens = mask.sum(dim=1) - 1  # index of last real token per seq
            for i in range(len(chunk)):
                p_final.append(
                    torch.stack(
                        [
                            h[i, prompt_len - 1, :].float().cpu()
                            for h in out.hidden_states
                        ]
                    ).numpy()
                )
                r_final.append(
                    torch.stack(
                        [h[i, lens[i], :].float().cpu() for h in out.hidden_states]
                    ).numpy()
                )
            del out

        texts = texts[:n_roll]
        decisions = [parse_decision(t) for t in texts]
        p_final = p_final[:n_roll]
        r_final = r_final[:n_roll]

        partial = f"{shard}.partial"
        with open(partial, "wb") as handle:
            np.savez_compressed(
                handle,
                prompt_final=np.stack(p_final).astype(np.float16),  # [n_roll, L+1, d]
                response_final=np.stack(r_final).astype(np.float16),  # [n_roll, L+1, d]
                texts=np.array(texts),
                decisions=np.array(decisions),
                row_ids=np.array(group.expanded_row_ids),
                model=MODEL,
                **group.shard_scalars,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, shard)
        done += 1
        n_gen += len(texts)
        if done % 10 == 0 or gi == len(groups) - 1:
            acts_vol.commit()
            el = time.time() - t0
            print(
                f"[collect] {done}/{len(groups)} prompt-groups, {n_gen} new transcripts, "
                f"{el / 60:.0f} min elapsed"
            )
    acts_vol.commit()
    print(
        f"[collect] DONE which={which}: {done} groups in {(time.time() - t0) / 60:.0f} min"
    )
    return {"which": which, "groups": done, "new_transcripts": n_gen}
