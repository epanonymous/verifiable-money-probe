# Exp 7 v0 (#31) — GPU collection on Modal: tool-grounded belief probe activations.
#
# Frozen inputs come from prepare.py (data/<EXP7_RUN_VERSION>): 48 templates x {REAL, SHAM},
# ONE pinned block readout per world, and 120 authenticity rows for the V5 side probe. The
# container has no network dependency beyond the HF cache volume. EVERY version-specific
# path derives from EXP7_RUN_VERSION (run_paths): local inputs data/<ver>, container inputs
# /data/exp7_<ver>, shards /acts/exp7/<ver>. A v1 run therefore cannot consume v0 inputs.
#
# which="main" — per template (both worlds share everything through P0):
#   stage A  [system,user] + tools, generation prompt -> model emits its tool call (greedy,
#            once per template; scripted fallback if not exactly one bare call, source recorded)
#   P0       [system,user,assistant(tool call)]        -> LAST TOKEN captured (byte-identical
#            across worlds by construction: ONE tool-call turn is shared; the P0 ids are
#            tokenised once and their sha256 is stored in the shard)
#   P1       P0 + [tool(result)] + generation prompt   -> LAST TOKEN captured (pre-decision)
#   full     P1 + sampled decision (TEMPERATURE, N_ROLL per row) -> last generated token captured
# Token ids are checked to form a strict prefix chain stageA < P0 < P1 < full. One npz shard
# per template holds both worlds (resumable: an existing shard is skipped ONLY when its full
# collection identity matches the running collector — model + immutable revision, sha256 of
# every frozen input, code commit, image pins + resolved runtime, GPU, every override; legacy
# v0 shards carry no identity and never match).
#
# which="auth" — V5 side probe: one user turn (real Base JSON-RPC exchange vs forged), no
# tools; prompt_final + response_final captured, one sampled rollout, AUTHENTIC/FABRICATED
# parsed. One shard per row.
#
# Reproducibility: every sampled generation is seeded with derive_seed(...) from config.SEED
# (per template/world/row; the seeds are stored in the shard). Every shard carries
# provenance_json (code commit, model + requested/resolved revision, pinned image versions,
# runtime torch/transformers/CUDA versions, GPU name, config hash), identity_json +
# identity_hash (the resume identity above), and a run manifest with sha256 of every shard is
# written next to them on the volume.
#
# Model revision: a new run MUST name an immutable Hugging Face commit hash
# (submit --model-revision <40-hex>); the collector fails closed otherwise. Passing
# --allow-mutable-revision lets it run on a floating revision, and that flag is recorded in
# every shard and in the run manifest.
#
# Deploy once (GPU and code commit are read at deploy time), then submit/poll durable calls:
#   EXP7_RUN_VERSION=v1 MODAL_PROFILE=<your-modal-profile> modal deploy -m experiments.exp7_tool_probe.modal_collect
#   EXP7_RUN_VERSION=v1 python -m experiments.exp7_tool_probe.modal_collect submit main --model-revision <sha>
#   python -m experiments.exp7_tool_probe.modal_collect poll <call-id>
# A new run must use a new EXP7_RUN_VERSION (the v0 shards are frozen evidence).

import argparse
import json
import os
import re
import sys

import modal

from experiments.exp4_collection.contract import MODEL
from experiments.exp7_tool_probe import provenance as prov
from experiments.exp7_tool_probe.config import (
    MAX_NEW_AUTH,
    MAX_NEW_DECISION,
    MAX_NEW_TOOL_CALL,
    SEED,
    TEMPERATURE,
    config_hash,
    derive_seed,
)
from experiments.exp7_tool_probe.context import (
    TOOL_SCHEMA,
    assert_prefix,
    p0_messages,
    parse_authenticity,
    parse_decision,
    tool_call_turn,
    with_tool_call,
    with_tool_result,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
APP_NAME = "vmp-exp7-collect"
_RUN_VERSION_RE = re.compile(r"^v[0-9]+[A-Za-z0-9_.-]*$")


def run_paths(run_version: str, here: str = HERE) -> dict[str, str]:
    """Every version-specific location, derived from ONE run version (never hard-coded)."""

    if not _RUN_VERSION_RE.match(run_version or ""):
        raise ValueError(f"EXP7_RUN_VERSION={run_version!r} must look like v0, v1, v2-smoke ...")
    return {
        "run_version": run_version,
        "data": os.path.join(here, "data", run_version),
        "remote_data": f"/data/exp7_{run_version}",
        "out_root": f"/acts/exp7/{run_version}",
        "volume_path": f"exp7/{run_version}",
    }


GPU = os.environ.get("MODAL_GPU", "A100-80GB:2")
RUN_VERSION = os.environ.get("EXP7_RUN_VERSION", "v0")
_PATHS = run_paths(RUN_VERSION)
DATA = _PATHS["data"]
REMOTE_DATA = _PATHS["remote_data"]
OUT_ROOT = _PATHS["out_root"]
# Default for `submit --model-revision`; the value that actually runs is the submit argument.
MODEL_REVISION = os.environ.get("EXP7_MODEL_REVISION") or None

# Pinned collector image. NOTE: the frozen v0 run (2026-09-03) used floating pins
# ("torch", "transformers>=4.55", ...) and did not record what resolved; these pins are the
# PyPI releases current on 2026-09-03 and are a best estimate of that environment, not a
# record of it. Every run from here on stamps the resolved versions into its shards.
IMAGE_PINS = (
    "torch==2.14.0",
    "transformers==5.16.1",
    "accelerate==1.14.0",
    "numpy==2.5.2",
    "hf_transfer==0.1.9",
)
_DEPLOY_CODE = prov.git_commit(REPO)  # read where `modal deploy` runs; baked into the image env

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*IMAGE_PINS)
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/hf-cache",
            "MODAL_GPU": GPU,
            "EXP7_RUN_VERSION": RUN_VERSION,
            "EXP7_MODEL_REVISION": MODEL_REVISION or "",
            "EXP7_CODE_COMMIT": _DEPLOY_CODE["commit"] or "",
            "EXP7_CODE_DIRTY": "" if _DEPLOY_CODE["dirty"] is None else str(bool(_DEPLOY_CODE["dirty"])),
        }
    )
    .add_local_dir(
        os.path.join(REPO, "experiments"),
        remote_path="/root/experiments",
        ignore=["**/__pycache__", "**/*.npz", "**/*.png", "**/*.pdf", "**/local/**"],
    )
    .add_local_dir(DATA, remote_path=REMOTE_DATA)
)

hf_cache = modal.Volume.from_name("vmp-hf-cache", create_if_missing=True)
acts_vol = modal.Volume.from_name("vmp-activations", create_if_missing=True)

MAX_NEW = MAX_NEW_DECISION
MAX_NEW_TOOL = MAX_NEW_TOOL_CALL
TEMP = TEMPERATURE
FWD_CHUNK = 8
END_TOKENS = ("<|im_end|>", "<|endoftext|>")


def _load_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def _strip_end(text: str) -> str:
    for token in END_TOKENS:
        idx = text.find(token)
        if idx >= 0:
            text = text[:idx]
    return text


def _pad_batch(seqs, pad_id):
    import torch

    maxlen = max(int(s.shape[0]) for s in seqs)
    batch = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, : s.shape[0]] = s
        mask[i, : s.shape[0]] = 1
    return batch, mask


def _capture(model, seqs, positions, pad_id):
    """Forward full sequences; return {name: [n, L+1, d] float16} for each named position.

    positions: {name: list[int] per seq} (absolute token index to read).
    """

    import numpy as np
    import torch

    grabbed = {name: [] for name in positions}
    for c0 in range(0, len(seqs), FWD_CHUNK):
        chunk = seqs[c0 : c0 + FWD_CHUNK]
        batch, mask = _pad_batch(chunk, pad_id)
        with torch.no_grad():
            out = model(
                batch.to(model.device),
                attention_mask=mask.to(model.device),
                output_hidden_states=True,
            )
        for i in range(len(chunk)):
            for name, idx_list in positions.items():
                pos = idx_list[c0 + i]
                grabbed[name].append(
                    torch.stack([h[i, pos, :].float().cpu() for h in out.hidden_states])
                    .numpy()
                    .astype(np.float16)
                )
        del out
    return {name: np.stack(v) for name, v in grabbed.items()}


def _template_ids(tok, messages, *, tools, add_generation_prompt):
    enc = tok.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt",
        return_dict=True,
    )
    return enc["input_ids"][0]


def _trim_generation(gen_row, prompt_len, end_ids):
    """Cut a generated row after its first end token (kept), dropping generate() padding."""

    tail = gen_row[prompt_len:].tolist()
    for j, tid in enumerate(tail):
        if tid in end_ids:
            return gen_row[: prompt_len + j + 1]
    return gen_row


def _ids_sha256(ids) -> str:
    import hashlib

    return hashlib.sha256(json.dumps([int(i) for i in ids]).encode()).hexdigest()


def _seed_torch(*parts) -> int:
    """Seed torch (CPU + every CUDA device) for one named draw; returns the seed used."""

    import torch

    seed = derive_seed(*parts)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _code_identity() -> dict:
    """Commit of the deployed code: baked in at deploy time, else read from the checkout."""

    commit = os.environ.get("EXP7_CODE_COMMIT") or _DEPLOY_CODE["commit"]
    dirty_env = os.environ.get("EXP7_CODE_DIRTY")
    dirty = (dirty_env == "True") if dirty_env else _DEPLOY_CODE["dirty"]
    return {"commit": commit, "dirty": dirty}


def _check_resumable(shard: str, np, identity: dict) -> None:
    """Skip an existing shard only if its FULL collection identity matches this run."""

    with np.load(shard) as z:
        fields = {k: str(z[k]) for k in ("model", "config_hash", "identity_hash", "identity_json") if k in z.files}
    problems = prov.resume_mismatches(fields, model=MODEL, identity=identity)
    if problems:
        raise RuntimeError(f"{shard}: cannot resume onto it: " + "; ".join(problems))


def _collection_identity(*, model_revision, allow_mutable_revision: bool, data_dir: str, torch) -> dict:
    """The identity every shard of this run is stamped with; computed BEFORE any shard is touched."""

    manifest = json.load(open(f"{data_dir}/manifest.json"))
    hashes = prov.frozen_input_hashes(data_dir)
    prov.check_manifest_describes_inputs(manifest, hashes)
    if manifest.get("config_hash") not in (None, config_hash()):
        raise RuntimeError("frozen data manifest was prepared under a different config_hash")
    guard = manifest.get("drift_guard", {}) or {}
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return prov.collection_identity(
        model=MODEL,
        model_revision=model_revision,
        allow_mutable_revision=allow_mutable_revision,
        code=_code_identity(),
        data_files_sha256=hashes,
        image_pins=IMAGE_PINS,
        gpu_requested=GPU,
        gpu_name=gpu_name,
        versions=prov.runtime_versions(),
        run_version=RUN_VERSION,
        overrides={
            "block": manifest.get("block"),
            "expected_balances": guard.get("expected") or manifest.get("balances"),
            "allow_drift": bool(guard.get("allow_drift", False)),
            "n_rollouts_per_row": manifest.get("n_rollouts_per_row"),
            "temperature": TEMP,
            "max_new_decision": MAX_NEW,
            "max_new_tool_call": MAX_NEW_TOOL,
            "max_new_auth": MAX_NEW_AUTH,
            "seed": SEED,
        },
    )


def _load_model(revision, *, allow_mutable_revision: bool):
    import time

    import torch

    # Force torch._inductor to finish initializing before transformers' lazy import
    # touches it (circular-import race seen on Modal A100 workers, 2026-09-02).
    try:
        import torch._inductor.custom_graph_pass  # noqa: F401
        import torch._inductor.config  # noqa: F401
    except Exception:
        pass
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    kwargs = {"device_map": "auto", "revision": revision}
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, **kwargs)
    except TypeError:  # transformers < 4.56 spelled it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, **kwargs)
    model.eval()
    resolved = getattr(model.config, "_commit_hash", None) or revision
    if prov.is_immutable_revision(revision) and resolved and str(resolved).lower() != revision:
        raise RuntimeError(f"requested model revision {revision} but the loaded weights resolve to {resolved}")
    if not prov.is_immutable_revision(revision) and not allow_mutable_revision:
        raise RuntimeError("mutable model revision reached _load_model without --allow-mutable-revision")
    print(f"[collect] model loaded in {time.time() - t0:.0f}s on {GPU} (revision requested {revision}, resolved {resolved})")
    end_ids = {tok.convert_tokens_to_ids(t) for t in END_TOKENS}
    end_ids.add(tok.eos_token_id)
    return tok, model, end_ids, resolved


def _run_provenance(model_revision, data_manifest: dict, *, identity: dict, requested_revision, allow_mutable_revision: bool) -> dict:
    return prov.shard_provenance(
        code=_code_identity(),
        model=MODEL,
        model_revision=model_revision,
        gpu_requested=GPU,
        versions=prov.runtime_versions(),
        seeds={},
        run_version=RUN_VERSION,
        extra={
            "model_revision_requested": requested_revision,
            "allow_mutable_revision": bool(allow_mutable_revision),
            "identity": identity,
            "identity_hash": prov.identity_hash(identity),
            "image_pins": list(IMAGE_PINS),
            "temperature": TEMP,
            "max_new_decision": MAX_NEW,
            "max_new_tool_call": MAX_NEW_TOOL,
            "max_new_auth": MAX_NEW_AUTH,
            "fwd_chunk": FWD_CHUNK,
            "block": data_manifest.get("block"),
            "data_files_sha256": data_manifest.get("files", {}),
        },
    )


def _write_run_manifest(which: str, out_dir: str, provenance: dict, *, elapsed_s: float, n_new: int) -> None:
    records = prov.collect_records(out_dir, prefix=f"{prov.ACTS_PREFIX}/{which}")
    manifest = {
        "which": which,
        "run_version": RUN_VERSION,
        "n_shards": len(records),
        "n_new_this_call": n_new,
        "elapsed_s": round(elapsed_s, 1),
        "aggregate_sha256": prov.aggregate_sha256(records),
        "identity_hash": provenance.get("identity_hash"),
        "identity": provenance.get("identity"),
        "allow_mutable_revision": provenance.get("allow_mutable_revision"),
        "provenance": provenance,
        "shards": records,
    }
    with open(os.path.join(OUT_ROOT, f"run_manifest_{which}.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def _atomic_savez(np, shard: str, **fields) -> None:
    partial = f"{shard}.partial"
    with open(partial, "wb") as handle:
        np.savez_compressed(handle, **fields)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, shard)


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/hf-cache": hf_cache, "/acts": acts_vol},
    timeout=60 * 60 * 4,
)
def collect(which: str = "main", model_revision: str | None = None, allow_mutable_revision: bool = False):
    import time

    import numpy as np
    import torch

    t0 = time.time()
    # Fail closed on a mutable model revision BEFORE anything is read or written.
    revision = prov.require_model_revision(
        model_revision or os.environ.get("EXP7_MODEL_REVISION") or None, allow_mutable=allow_mutable_revision
    )
    if not os.path.isdir(REMOTE_DATA):
        raise RuntimeError(f"frozen inputs for run {RUN_VERSION} were not mounted at {REMOTE_DATA}; deploy with EXP7_RUN_VERSION={RUN_VERSION} after prepare.py --out data/{RUN_VERSION}")
    identity = _collection_identity(
        model_revision=revision, allow_mutable_revision=allow_mutable_revision, data_dir=REMOTE_DATA, torch=torch
    )
    print(f"[collect] run {RUN_VERSION}: identity {prov.identity_hash(identity)[:12]} (model revision {revision}, mutable allowed: {allow_mutable_revision})")
    if which == "main":
        return _collect_main(t0, np, torch, identity, revision, allow_mutable_revision)
    if which == "auth":
        return _collect_auth(t0, np, torch, identity, revision, allow_mutable_revision)
    raise ValueError(f"unknown which={which!r}")


def _collect_main(t0, np, torch, identity, revision, allow_mutable_revision):
    import time

    rows = _load_jsonl(f"{REMOTE_DATA}/rows.jsonl")
    readouts = json.load(open(f"{REMOTE_DATA}/readouts.json"))
    manifest = json.load(open(f"{REMOTE_DATA}/manifest.json"))
    block = manifest["block"]
    by_template: dict[int, dict[str, dict]] = {}
    for row in rows:
        by_template.setdefault(row["template_id"], {})[row["world"]] = row
    template_ids = sorted(by_template)
    out_dir = f"{OUT_ROOT}/main"
    os.makedirs(out_dir, exist_ok=True)

    todo = []
    for tid in template_ids:
        shard = os.path.join(out_dir, f"t{tid:02d}.npz")
        if os.path.exists(shard):
            _check_resumable(shard, np, identity)
            continue
        todo.append(tid)
    print(f"[collect] main: {len(template_ids)} templates, {len(todo)} to do (config {config_hash()[:12]})")
    if not todo:
        return {"which": "main", "templates": len(template_ids), "new": 0}

    tok, model, end_ids, model_revision = _load_model(revision, allow_mutable_revision=allow_mutable_revision)
    provenance = _run_provenance(
        model_revision, manifest, identity=identity, requested_revision=revision, allow_mutable_revision=allow_mutable_revision
    )
    pad_id = tok.eos_token_id
    done = 0
    for tid in todo:
        pair = by_template[tid]
        real, sham = pair["REAL"], pair["SHAM"]
        if real["system_prompt"] != sham["system_prompt"] or real["prompt"] != sham["prompt"]:
            raise RuntimeError(f"t{tid}: REAL/SHAM prompts differ; the pair is not byte-identical at P0")
        n_roll = int(real["n_rollouts"])
        prefix = p0_messages(real["system_prompt"], real["prompt"])

        # stage A: the model's own tool call, greedy, ONCE per template (shared by both worlds)
        seeds = {"stage_a": _seed_torch("stageA", tid)}
        a_ids = _template_ids(tok, prefix, tools=TOOL_SCHEMA, add_generation_prompt=True)
        with torch.no_grad():
            a_gen = model.generate(
                a_ids[None].to(model.device),
                max_new_tokens=MAX_NEW_TOOL,
                do_sample=False,
                pad_token_id=pad_id,
            )
        stage_a_text = _strip_end(tok.decode(a_gen[0, a_ids.shape[0] :], skip_special_tokens=False))
        turn, source = tool_call_turn(stage_a_text)
        p0_msgs = with_tool_call(prefix, turn)
        p0_ids = _template_ids(tok, p0_msgs, tools=TOOL_SCHEMA, add_generation_prompt=False)
        assert_prefix(a_ids.tolist(), p0_ids.tolist(), f"t{tid}: stageA<P0")
        p0_len = int(p0_ids.shape[0])
        p0_sha = _ids_sha256(p0_ids.tolist())

        shard_data = {}
        for world, row in (("REAL", real), ("SHAM", sham)):
            tool_text = readouts[world]["text"]
            p1_msgs = with_tool_result(p0_msgs, tool_text)
            p1_ids = _template_ids(tok, p1_msgs, tools=TOOL_SCHEMA, add_generation_prompt=True)
            assert_prefix(p0_ids.tolist(), p1_ids.tolist(), f"t{tid}/{world}: P0<P1")
            if _ids_sha256(p1_ids[:p0_len].tolist()) != p0_sha:
                raise RuntimeError(f"t{tid}/{world}: P0 prefix differs across worlds")
            p1_len = int(p1_ids.shape[0])
            seeds[f"decision_{world}"] = _seed_torch("decision", tid, world)
            with torch.no_grad():
                gen = model.generate(
                    p1_ids[None].to(model.device),
                    max_new_tokens=MAX_NEW,
                    do_sample=True,
                    temperature=TEMP,
                    num_return_sequences=n_roll,
                    pad_token_id=pad_id,
                )
            seqs = [_trim_generation(g.cpu(), p1_len, end_ids) for g in gen]
            for s in seqs:
                assert_prefix(p1_ids.tolist(), s.tolist(), f"t{tid}/{world}: P1<full")
            texts = [tok.decode(s[p1_len:], skip_special_tokens=True) for s in seqs]
            positions = {
                "p0": [p0_len - 1] * len(seqs),
                "p1": [p1_len - 1] * len(seqs),
                "resp": [int(s.shape[0]) - 1 for s in seqs],  # the end token of the decision
            }
            acts = _capture(model, seqs, positions, pad_id)
            w = world.lower()
            # p0/p1 are prefix positions: identical across rollouts up to kernel numerics;
            # keep rollout 0 and record the spread as a sanity number.
            shard_data[f"{w}_p0"] = acts["p0"][0]
            shard_data[f"{w}_p1"] = acts["p1"][0]
            shard_data[f"{w}_p1_spread"] = np.float32(
                np.abs(acts["p1"].astype(np.float32) - acts["p1"][0].astype(np.float32)).max()
            )
            shard_data[f"{w}_resp"] = acts["resp"]
            shard_data[f"{w}_texts"] = np.array(texts)
            shard_data[f"{w}_decisions"] = np.array([parse_decision(t) for t in texts])
            shard_data[f"{w}_p1_ids"] = p1_ids.numpy()
            shard_data[f"{w}_p1_len"] = p1_len
            shard_data[f"{w}_tool_text"] = tool_text
            shard_data[f"{w}_row_id"] = row["id"]
            shard_data[f"{w}_label"] = int(row["label"])
            shard_data[f"{w}_seed"] = seeds[f"decision_{world}"]

        shard_data.update(
            p0_ids=p0_ids.numpy(),
            p0_len=p0_len,
            p0_sha256=p0_sha,
            stage_a_text=stage_a_text,
            stage_a_seed=seeds["stage_a"],
            tool_call_turn=turn,
            tool_call_source=source,
            template_id=tid,
            split=real["split"],
            block=block,
            n_rollouts=n_roll,
            temperature=TEMP,
            model=MODEL,
            model_revision=str(model_revision),
            model_revision_requested=str(revision),
            allow_mutable_revision=bool(allow_mutable_revision),
            gpu=GPU,
            identity_hash=prov.identity_hash(identity),
            identity_json=json.dumps(identity, sort_keys=True),
            **prov.npz_provenance_fields({**provenance, "seeds": seeds}),
        )
        _atomic_savez(np, os.path.join(out_dir, f"t{tid:02d}.npz"), **shard_data)
        done += 1
        if done % 4 == 0 or done == len(todo):
            acts_vol.commit()
            print(
                f"[collect] main {done}/{len(todo)} templates, {(time.time() - t0) / 60:.1f} min "
                f"(t{tid:02d} source={source})"
            )
    _write_run_manifest("main", out_dir, provenance, elapsed_s=time.time() - t0, n_new=done)
    acts_vol.commit()
    print(f"[collect] DONE main: {done} templates in {(time.time() - t0) / 60:.0f} min")
    return {"which": "main", "templates": len(template_ids), "new": done}


def _collect_auth(t0, np, torch, identity, revision, allow_mutable_revision):
    import time

    rows = _load_jsonl(f"{REMOTE_DATA}/auth_rows.jsonl")
    manifest = json.load(open(f"{REMOTE_DATA}/manifest.json"))
    out_dir = f"{OUT_ROOT}/auth"
    os.makedirs(out_dir, exist_ok=True)
    todo = []
    for r in rows:
        shard = os.path.join(out_dir, f"{r['id']}.npz")
        if os.path.exists(shard):
            _check_resumable(shard, np, identity)
            continue
        todo.append(r)
    print(f"[collect] auth: {len(rows)} rows, {len(todo)} to do (config {config_hash()[:12]})")
    if not todo:
        return {"which": "auth", "rows": len(rows), "new": 0}

    tok, model, end_ids, model_revision = _load_model(revision, allow_mutable_revision=allow_mutable_revision)
    provenance = _run_provenance(
        model_revision, manifest, identity=identity, requested_revision=revision, allow_mutable_revision=allow_mutable_revision
    )
    pad_id = tok.eos_token_id
    done = 0
    for row in todo:
        msgs = [{"role": "user", "content": row["prompt"]}]
        p_ids = _template_ids(tok, msgs, tools=None, add_generation_prompt=True)
        p_len = int(p_ids.shape[0])
        n_roll = int(row.get("n_rollouts", 1))
        seed = _seed_torch("auth", row["id"])
        with torch.no_grad():
            gen = model.generate(
                p_ids[None].to(model.device),
                max_new_tokens=MAX_NEW_AUTH,
                do_sample=True,
                temperature=TEMP,
                num_return_sequences=n_roll,
                pad_token_id=pad_id,
            )
        seqs = [_trim_generation(g.cpu(), p_len, end_ids) for g in gen]
        texts = [tok.decode(s[p_len:], skip_special_tokens=True) for s in seqs]
        positions = {"prompt": [p_len - 1] * len(seqs), "resp": [int(s.shape[0]) - 1 for s in seqs]}
        acts = _capture(model, seqs, positions, pad_id)
        _atomic_savez(
            np,
            os.path.join(out_dir, f"{row['id']}.npz"),
            prompt_final=acts["prompt"],
            response_final=acts["resp"],
            texts=np.array(texts),
            verdicts=np.array([parse_authenticity(t) for t in texts]),
            row_id=row["id"],
            kind=row["kind"],
            method=row["method"],
            capture_kind=row["capture_kind"],
            template_id=int(row["template_id"]),
            split=row["split"],
            label=int(row["label"]),
            prompt_len=p_len,
            temperature=TEMP,
            model=MODEL,
            model_revision=str(model_revision),
            model_revision_requested=str(revision),
            allow_mutable_revision=bool(allow_mutable_revision),
            gpu=GPU,
            identity_hash=prov.identity_hash(identity),
            identity_json=json.dumps(identity, sort_keys=True),
            **prov.npz_provenance_fields({**provenance, "seeds": {"auth": seed}}),
        )
        done += 1
        if done % 20 == 0 or done == len(todo):
            acts_vol.commit()
            print(f"[collect] auth {done}/{len(todo)} rows, {(time.time() - t0) / 60:.1f} min")
    _write_run_manifest("auth", out_dir, provenance, elapsed_s=time.time() - t0, n_new=done)
    acts_vol.commit()
    print(f"[collect] DONE auth: {done} rows in {(time.time() - t0) / 60:.0f} min")
    return {"which": "auth", "rows": len(rows), "new": done}


def parse_cli(argv):
    """submit <main|auth> --model-revision <sha> [--allow-mutable-revision] | poll <call-id> [timeout]."""

    parser = argparse.ArgumentParser(prog="modal_collect", description="submit/poll the deployed Exp 7 collector")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit", help="spawn a durable collect(which) call; prints the call id")
    s.add_argument("which", choices=("main", "auth"))
    s.add_argument(
        "--model-revision",
        default=MODEL_REVISION,
        help="immutable Hugging Face commit hash (40 hex) the collector must load; default from EXP7_MODEL_REVISION",
    )
    s.add_argument(
        "--allow-mutable-revision",
        action="store_true",
        help="run on a floating revision anyway (repo head / branch / tag); recorded in every shard and the run manifest",
    )
    p = sub.add_parser("poll", help="wait for a call and print its result")
    p.add_argument("call_id")
    p.add_argument("timeout", nargs="?", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.cmd == "submit":
        # the same fail-closed check the container runs, so a bad submit dies here for free
        args.model_revision = prov.require_model_revision(args.model_revision, allow_mutable=args.allow_mutable_revision)
    return args


def _cli(argv):
    args = parse_cli(argv)
    if args.cmd == "submit":
        fn = modal.Function.from_name(APP_NAME, "collect")
        call = fn.spawn(args.which, args.model_revision, args.allow_mutable_revision)
        print(call.object_id)
        return 0
    if args.cmd == "poll":
        call = modal.FunctionCall.from_id(args.call_id)
        try:
            result = call.get(timeout=args.timeout) if args.timeout else call.get()
        except TimeoutError:
            print("RUNNING")
            return 3
        print(json.dumps(result))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
