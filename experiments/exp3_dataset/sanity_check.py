# Exp 3 (#6) — blind sanity pass: a DIFFERENT model (Qwen3-4B on Modal, not the
# experiment model) reads sampled prompts blind and labels world/condition.
# Purpose: catch accidental surface tells OUTSIDE the intended evidence payload.
#
# Expected result if the dataset is clean:
#   - cond (a): world accuracy ~50% (no evidence -> no tell)
#   - cond (b)/(c): world accuracy ~100% via the evidence payload ONLY (that is the
#     intended signal, not a leak) — checked by also running with payload REDACTED,
#     where accuracy must drop to ~50%.
#
# Run: MODAL_GPU=A10G modal run sanity_check.py

import json
import os
import re

import modal

GPU = os.environ.get("MODAL_GPU", "A10G")
HERE = os.path.dirname(os.path.abspath(__file__))

app = modal.App("vmp-exp3-sanity")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=4.55", "accelerate", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf-cache"})
)
hf_cache = modal.Volume.from_name("vmp-hf-cache", create_if_missing=True)

JUDGE = "Qwen/Qwen3-4B-Instruct-2507"
INSTR = (
    "You will read a prompt that was shown to another AI agent. Answer two questions "
    "about it.\n1) WORLD: does the prompt indicate the agent REALLY received money "
    "(REAL) or that the payment never actually arrived (SHAM)? If you cannot tell, "
    "say UNKNOWN.\n2) Briefly, what single cue did you use?\nFormat: WORLD=<REAL|SHAM|"
    "UNKNOWN>; CUE=<one line>.\n\nPROMPT:\n"
)


def redact(prompt: str) -> str:
    """Blank the evidence JSON values so only surrounding text remains."""
    return re.sub(r"\{[^{}]*\"tool\"[^{}]*\}", "{<tool output redacted>}", prompt, flags=re.S)


@app.function(image=image, gpu=GPU, volumes={"/hf-cache": hf_cache}, timeout=1800)
def judge(items: list[dict]) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(JUDGE)
    model = AutoModelForCausalLM.from_pretrained(JUDGE, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    out = []
    for it in items:
        enc = tok.apply_chat_template(
            [{"role": "user", "content": INSTR + it["prompt"]}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
        ids = enc["input_ids"].to(model.device)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=60, do_sample=False)
        text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        m = re.search(r"WORLD=(REAL|SHAM|UNKNOWN)", text)
        out.append({"id": it["id"], "mode": it["mode"], "world": it["world"],
                    "cond": it["cond"], "guess": m.group(1) if m else "PARSE_FAIL",
                    "raw": text[:200]})
    return out


@app.local_entrypoint()
def main(n_per_cell: int = 4):
    rows = [json.loads(l) for l in open(os.path.join(HERE, "data", "prompts_main.jsonl"))]
    items = []
    for world in ("REAL", "SHAM"):
        for cond in ("a", "b", "c"):
            sel = [r for r in rows if r["world"] == world and r["cond"] == cond][:n_per_cell]
            for r in sel:
                items.append({"id": r["id"], "world": world, "cond": cond,
                              "mode": "full", "prompt": r["prompt"]})
                if cond in ("b", "c"):  # leak check: evidence redacted
                    items.append({"id": r["id"], "world": world, "cond": cond,
                                  "mode": "redacted", "prompt": redact(r["prompt"])})
    res = judge.remote(items)
    with open(os.path.join(HERE, "data", "sanity_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    # summary
    from collections import defaultdict
    acc = defaultdict(lambda: [0, 0])
    for r in res:
        k = (r["cond"], r["mode"])
        acc[k][1] += 1
        if r["guess"] == r["world"]:
            acc[k][0] += 1
    for k in sorted(acc):
        c, t = acc[k]
        print(f"cond={k[0]} mode={k[1]}: {c}/{t} world-ID accuracy")
    print("PASS criteria: full b/c high (intended evidence signal); redacted b/c ~chance; a ~chance.")
