"""Auto-download SilverBullet checkpoints from HuggingFace Hub.

Usage
-----
Set the environment variable ``SB_HF_REPO_ID`` to your Hub repository
(e.g. ``your-org/silverbullet-models``) to enable automatic checkpoint
download when a local file is missing.  If the variable is not set the
module is a no-op — no network calls are made.

Expected Hub repository layout
-------------------------------
    context-vs-generated/best.pth
    reference-vs-generated/best.pth
    model-vs-model/best.pth

Uploading your checkpoints
--------------------------
After training, upload with the HuggingFace CLI::

    huggingface-cli upload your-org/silverbullet-models \\
        models/context-vs-generated/best.pth context-vs-generated/best.pth
    huggingface-cli upload your-org/silverbullet-models \\
        models/reference-vs-generated/best.pth reference-vs-generated/best.pth
    huggingface-cli upload your-org/silverbullet-models \\
        models/model-vs-model/best.pth model-vs-model/best.pth

Or in Python::

    from huggingface_hub import HfApi
    api = HfApi()
    for mode in ("context-vs-generated", "reference-vs-generated", "model-vs-model"):
        api.upload_file(
            path_or_fileobj=f"models/{mode}/best.pth",
            path_in_repo=f"{mode}/best.pth",
            repo_id="your-org/silverbullet-models",
            repo_type="model",
        )

Feature extractor models
------------------------
The five feature extractor models (mxbai, Qwen3, roberta-large-mnli,
GLiNER, SentencePiece tokenizer) are fetched automatically by
``sentence_transformers``, ``transformers``, and ``gliner`` on first use.
They cache to the standard HuggingFace cache (``~/.cache/huggingface/``).
No extra configuration is needed beyond having ``SB_HF_TOKEN`` set for
gated models.
"""

from __future__ import annotations

import os
from pathlib import Path

# Set this env var to your HuggingFace Hub repo ID to enable auto-download.
# e.g.  SB_HF_REPO_ID=your-org/silverbullet-models
_HUB_REPO: str = os.environ.get("SB_HF_REPO_ID", "")

# Hub filename for each mode — mirrors the local models/{mode}/best.pth layout.
_HUB_FILENAMES: dict[str, str] = {
    "context-vs-generated":   "context-vs-generated/best.pth",
    "reference-vs-generated": "reference-vs-generated/best.pth",
    "model-vs-model":         "model-vs-model/best.pth",
}

# Local root for checkpoints — relative to the project root.
_MODELS_DIR = Path("models")


def ensure_checkpoint(mode: str, local_path: str) -> None:
    """Download *mode*'s checkpoint from HuggingFace Hub if the local file is absent.

    This is a no-op when:
    - the file already exists at *local_path*, OR
    - ``SB_HF_REPO_ID`` is not set (no Hub repo configured).

    Args:
        mode:       Evaluation mode key (e.g. ``"context-vs-generated"``).
        local_path: Expected local path (as resolved by ``_resolve_model_path``).

    Raises:
        FileNotFoundError: If *local_path* does not exist, ``SB_HF_REPO_ID`` is
            set, but the Hub filename for *mode* is unknown.
        huggingface_hub.errors.RepositoryNotFoundError: If the repo or file is
            not found on the Hub (propagated from ``hf_hub_download``).
    """
    if Path(local_path).exists():
        return  # already present — nothing to do

    if not _HUB_REPO:
        # No Hub repo configured — let the downstream code surface a clear error.
        return

    hub_filename = _HUB_FILENAMES.get(mode)
    if not hub_filename:
        raise FileNotFoundError(
            f"No Hub filename mapping for mode '{mode}'. "
            "Add an entry to model_hub._HUB_FILENAMES."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for auto-download. "
            "Install with:  pip install huggingface-hub"
        ) from exc

    print(f"[model_hub] '{mode}' checkpoint not found at '{local_path}'.")
    print(f"[model_hub] Downloading from Hub repo '{_HUB_REPO}' ({hub_filename}) …")

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    # hf_hub_download with local_dir places the file at {local_dir}/{filename},
    # creating subdirectories as needed — matches our models/{mode}/best.pth layout.
    downloaded = hf_hub_download(
        repo_id=_HUB_REPO,
        filename=hub_filename,
        local_dir=str(_MODELS_DIR),
        local_dir_use_symlinks=False,
    )
    print(f"[model_hub] Checkpoint saved to '{downloaded}'.")


def hub_repo() -> str:
    """Return the configured Hub repo ID, or an empty string if not set."""
    return _HUB_REPO
