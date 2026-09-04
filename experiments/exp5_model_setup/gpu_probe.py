# Probe which GPU classes this Modal workspace can use without a payment method.
import os

import modal

GPU = os.environ.get("MODAL_GPU", "T4")
app = modal.App("vmp-gpu-probe")


@app.function(gpu=GPU, image=modal.Image.debian_slim(), timeout=120)
def probe():
    import subprocess

    return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout


@app.local_entrypoint()
def main():
    print(f"GPU={GPU} ->", probe.remote())
