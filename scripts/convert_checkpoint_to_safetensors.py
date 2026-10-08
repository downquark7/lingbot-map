"""Convert a LingBot-Map torch checkpoint to .safetensors.

A .safetensors file contains only raw tensors, so loading it can never execute
code. Convert once, then point --model_path at the .safetensors file.

Usage:
    python scripts/convert_checkpoint_to_safetensors.py lingbot-map.pt
    python scripts/convert_checkpoint_to_safetensors.py lingbot-map.pt --trust_checkpoint
"""

import argparse
import os

import torch
from safetensors.torch import save_file

from lingbot_map.utils.checkpoint import load_checkpoint_state_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to the .pt/.pth checkpoint")
    parser.add_argument("--output", default=None, help="Output path (default: <input>.safetensors)")
    parser.add_argument("--trust_checkpoint", action="store_true",
                        help="Allow full unpickling if the file is not a plain-tensor checkpoint")
    args = parser.parse_args()

    output = args.output or os.path.splitext(args.input)[0] + ".safetensors"
    state_dict = load_checkpoint_state_dict(args.input, "cpu", args.trust_checkpoint)

    # safetensors rejects non-tensors and tensors that share storage; clone breaks sharing.
    tensors = {k: v.detach().contiguous().clone() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    skipped = sorted(set(state_dict) - set(tensors))
    if skipped:
        print(f"Skipping {len(skipped)} non-tensor entries: {skipped[:5]}{' ...' if len(skipped) > 5 else ''}")

    save_file(tensors, output)
    print(f"Wrote {len(tensors)} tensors to {output}")


if __name__ == "__main__":
    main()
