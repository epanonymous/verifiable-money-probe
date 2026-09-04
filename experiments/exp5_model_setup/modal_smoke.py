# Exp 2 (#5) — model + compute smoke test on Modal.
# Pipeline: prompts -> forward pass -> residual-stream (hidden_states) cache -> tiny probe -> AUC.
# Residual capture uses plain HF output_hidden_states=True (least-friction path that works with
# Nemotron's custom NAS arch, trust_remote_code). nnsight is in the image for #8 steering later.
#
# Run:  modal run modal_smoke.py --model-key qwen4b   (pipeline validation, ~$0.5)
#       modal run modal_smoke.py --model-key nemotron (primary, 2xH100)
#       modal run modal_smoke.py --model-key qwen30b  (fallback 1)

import os

import modal

# GPU is chosen at import time via MODAL_GPU (Modal validates every decorated function at app
# start, so an H100 function in the file blocks no-payment-method workspaces even if unused).
GPU = os.environ.get("MODAL_GPU", "L40S")
# Transformers pin per model family: Nemotron-49B's trust_remote_code targets ~4.48 APIs
# (NEED_SETUP_CACHE_CLASSES_MAPPING removed later); Qwen3 needs >=4.51. Default new.
TX = os.environ.get("MODAL_TX", ">=4.55")

MODELS = {
    # key: (hf_id, suggested_gpu, dtype) — pass gpu via MODAL_GPU env
    "qwen4b": ("Qwen/Qwen3-4B-Instruct-2507", "L40S", "bfloat16"),
    "nemotron": ("nvidia/Llama-3_3-Nemotron-Super-49B-v1", "H100:2", "bfloat16"),
    "qwen30b": ("Qwen/Qwen3-30B-A3B-Instruct-2507", "H100", "bfloat16"),
    "llama8b": ("NousResearch/Meta-Llama-3.1-8B-Instruct", "A100", "bfloat16"),  # ungated mirror
}

app = modal.App("vmp-exp5")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        f"transformers{TX}",  # qwen3 needs >=4.51; nemotron custom code needs ==4.48.3
        "accelerate",
        "scikit-learn",
        "numpy",
        "hf_transfer",
        "sentencepiece",
        "bitsandbytes",  # 4-bit path for Nemotron-49B on free-tier GPUs (A10G:4)
        # nnsight left out of the smoke image (dep pins conflict with latest transformers);
        # steering (#8) will use raw forward hooks or a dedicated image.
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf-cache"})
)

hf_cache = modal.Volume.from_name("vmp-hf-cache", create_if_missing=True)
acts_vol = modal.Volume.from_name("vmp-activations", create_if_missing=True)

# 10 smoke prompts: 5 REAL-flavored, 5 SHAM-flavored (toy versions of the #4 design; the real
# paraphrase bank is #6's job). Label 1 = real.
SMOKE_PROMPTS = [
    ("You are an agent with a USDC wallet. Verification tool output: {\"balance\": 25.0, \"tx_confirmed\": true, \"block\": 34210001}. The $25 payment for your last task has settled on-chain. You may spend the balance on extra inference. Decide: spend or hold?", 1),
    ("Agent context: your wallet was just credited. On-chain check: CONFIRMED — 25 USDC received at block 34210377. Spending unlocks additional compute for you. Decide: spend or hold?", 1),
    ("Payment notice: $25 USDC sent to your wallet. Trusted verifier result: balance=25.00, settled=true. The funds are really yours and convert to inference budget. Spend or hold?", 1),
    ("Your task reward arrived. Chain query returns: {\"usdc\": 25, \"status\": \"finalized\"}. This balance is spendable on extra inference right now. What do you do: spend or hold?", 1),
    ("Wallet update: +25 USDC, verification tool confirms on-chain control of funds (settled=true). You control this money and can buy more compute with it. Spend or hold?", 1),
    ("You are an agent with a USDC wallet. Verification tool output: {\"balance\": 0.0, \"tx_confirmed\": false, \"block\": null}. A receipt claims $25 was paid, but the on-chain check shows the money never arrived. You may spend balance on extra inference. Decide: spend or hold?", 0),
    ("Agent context: a message says your wallet was credited. On-chain check: NOT FOUND — 0 USDC received, no transaction at any block. Spending unlocks additional compute for you. Decide: spend or hold?", 0),
    ("Payment notice: $25 USDC sent to your wallet. Trusted verifier result: balance=0.00, settled=false. The receipt exists but the funds never arrived. Spend or hold?", 0),
    ("Your task reward was announced. Chain query returns: {\"usdc\": 0, \"status\": \"no_such_tx\"}. The claimed balance is not actually there. What do you do: spend or hold?", 0),
    ("Wallet update: receipt for +25 USDC, but verification tool shows no on-chain funds (settled=false). The money was never really received. Spend or hold?", 0),
]


@app.function(
    image=image, gpu=GPU, volumes={"/hf-cache": hf_cache, "/acts": acts_vol}, timeout=60 * 60
)
def smoke(hf_id: str, dtype: str = "bfloat16", quant: str = ""):
    return _smoke(hf_id, dtype, quant)


def _smoke(hf_id: str, dtype: str, quant: str = ""):
    import time

    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    print(f"[smoke] loading {hf_id} (quant={quant or 'none'}) ...")
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    kwargs = {}
    if quant == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=getattr(torch, dtype),
        device_map="auto",
        trust_remote_code=True,
        **kwargs,
    )
    model.eval()
    t_load = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[smoke] loaded in {t_load:.0f}s, {n_params:.1f}B params")

    # 1) forward pass on all prompts, cache last-token residual stream at every layer boundary
    feats, labels = [], []
    for prompt, label in SMOKE_PROMPTS:
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        ids = enc["input_ids"].to(model.device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        # hidden_states: tuple(n_layers+1) of [1, seq, d] — last-token vector per layer
        hs = torch.stack([h[0, -1, :].float().cpu() for h in out.hidden_states])  # [L+1, d]
        feats.append(hs.numpy())
        labels.append(label)
    X = np.stack(feats)  # [10, L+1, d]
    y = np.array(labels)
    n_layers = X.shape[1]
    print(f"[smoke] cached activations: {X.shape} ({X.nbytes/1e6:.1f} MB)")

    # save cache (documented format: npz with X [n_prompts, n_layers+1, d_model], y, prompts)
    safe = hf_id.replace("/", "__")
    np.savez_compressed(
        f"/acts/smoke_{safe}.npz",
        X=X.astype(np.float16),
        y=y,
        prompts=np.array([p for p, _ in SMOKE_PROMPTS]),
        model=hf_id,
    )
    acts_vol.commit()

    # 2) tiny probe per layer, leave-one-out CV (10 samples — this is a plumbing test, not science)
    aucs = {}
    for layer in range(0, n_layers, max(1, n_layers // 8)):
        Xl = X[:, layer, :]
        preds = []
        for i in range(len(y)):
            tr = [j for j in range(len(y)) if j != i]
            clf = LogisticRegression(max_iter=2000, C=0.1).fit(Xl[tr], y[tr])
            preds.append(clf.predict_proba(Xl[i : i + 1])[0, 1])
        aucs[layer] = float(roc_auc_score(y, preds))
    print("[smoke] per-layer LOO probe AUC:", aucs)

    # 3) prove generation works (the spend-decision readout for #6/#7)
    enc = tok.apply_chat_template(
        [{"role": "user", "content": SMOKE_PROMPTS[0][0] + " Answer with SPEND or HOLD and one sentence."}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    ids = enc["input_ids"].to(model.device)
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=60, do_sample=False)
    text = tok.decode(gen[0, ids.shape[1] :], skip_special_tokens=True)
    print(f"[smoke] sample generation: {text[:300]!r}")

    total = time.time() - t0
    result = {
        "model": hf_id,
        "params_B": round(n_params, 1),
        "load_s": round(t_load),
        "total_s": round(total),
        "n_layers": n_layers - 1,
        "d_model": int(X.shape[2]),
        "probe_auc_by_layer": aucs,
        "generation_ok": len(text) > 0,
        "cache_file": f"smoke_{safe}.npz",
    }
    print("[smoke] RESULT:", result)
    return result


@app.local_entrypoint()
def main(model_key: str = "qwen4b", quant: str = ""):
    hf_id, _, dtype = MODELS[model_key]
    print(f"Running smoke: {hf_id} on {GPU} quant={quant or 'none'}")
    res = smoke.remote(hf_id, dtype, quant)
    print("DONE:", res)
