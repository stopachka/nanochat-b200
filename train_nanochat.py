import modal

NANOCHAT_REPO = "https://github.com/stopachka/nanochat-b200"
WORKDIR = "/root/nanochat"
CACHE_DIR = "/root/.cache/nanochat"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "curl", "build-essential")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        f"git clone {NANOCHAT_REPO} {WORKDIR}",
        f"cd {WORKDIR} && /root/.local/bin/uv venv && /root/.local/bin/uv sync --extra gpu",
    )
    .env({"PATH": "/root/.local/bin:/root/nanochat/.venv/bin:${PATH}"})
)

app = modal.App("nanochat-speedrun")
volume = modal.Volume.from_name("nanochat-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="B200!:8",
    volumes={CACHE_DIR: volume},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=8 * 60 * 60,
)
def train():
    import subprocess

    subprocess.run(
        ["bash", "runs/speedrun.sh"],
        cwd=WORKDIR,
        env={
            **__import__("os").environ,
            "WANDB_RUN": "speedrun",
            "NANOCHAT_BASE_DIR": CACHE_DIR,
        },
        check=True,
    )
    volume.commit()


@app.local_entrypoint()
def main():
    train.remote()
