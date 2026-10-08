"""Checkpoint loading that does not execute arbitrary pickle code by default.

``torch.load(..., weights_only=False)`` unpickles the file, and a crafted
pickle can run any Python on your machine. ``weights_only=True`` only accepts
tensors and plain containers, which is all a model state dict needs.
``.safetensors`` files are always safe to load.
"""

import pickle

import torch

TRUST_HINT = (
    "Only do this for files from a source you trust (e.g. the official "
    "robbyant/lingbot-map Hugging Face repo): rerun with --trust_checkpoint, or "
    "convert it once with scripts/convert_checkpoint_to_safetensors.py --trust_checkpoint "
    "and load the resulting .safetensors file from then on."
)


def load_checkpoint_state_dict(path: str, map_location="cpu", trust_checkpoint: bool = False):
    """Load a model state dict from ``.safetensors`` or a torch ``.pt``/``.pth`` file.

    Torch checkpoints are first loaded with ``weights_only=True``. If that is
    rejected because the file holds arbitrary Python objects, the full
    (code-executing) unpickler is only used when ``trust_checkpoint`` is set.
    """
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(str(path), device=str(map_location))

    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as e:
        if not trust_checkpoint:
            raise RuntimeError(
                f"Refusing to fully unpickle {path}: it contains Python objects beyond "
                f"plain tensors, and unpickling them can execute code. {TRUST_HINT}"
            ) from e
        print(f"WARNING: loading {path} with weights_only=False (--trust_checkpoint given).")
        ckpt = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt
