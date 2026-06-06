#!/usr/bin/env python3
"""Showcase: language-model perplexity on local WikiText-2.

Computes the perplexity of SmolLM2-135M over the bundled WikiText-2 test split
(``data/benchmark/wikitext``) loaded via :mod:`smlx.data.local`. Perplexity is
measured directly: the text is tokenised, split into fixed-length windows, and
the model's next-token cross-entropy is averaged over all predicted tokens --
the standard language-modelling metric, computed on in-repo data with no
download of the dataset.

Run::

    python examples/showcase/smollm2_wikitext_perplexity.py
    python examples/showcase/smollm2_wikitext_perplexity.py --seq-len 512 --max-sequences 40
"""

from __future__ import annotations

import argparse
import math

import mlx.core as mx
import mlx.nn as nn

from smlx.data import local
from smlx.models.SmolLM2_135M import load


def gather_text(min_chars: int) -> str:
    """Concatenate non-empty WikiText-2 test lines into one document."""
    ds = local.load("wikitext", split="test")
    parts: list[str] = []
    total = 0
    for row in ds:
        line = (row.get("text") or "").strip()
        if line:
            parts.append(line)
            total += len(line)
        if total >= min_chars:
            break
    return "\n\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--seq-len", type=int, default=512, help="Tokens per window")
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=30,
        help="Number of windows to evaluate (caps runtime)",
    )
    args = parser.parse_args()

    if not local.is_available("wikitext"):
        print(
            "WikiText not present. Fetch with: python -m smlx.tools.download_data --dataset wikitext"
        )
        return 1

    # Enough characters to fill the requested number of windows comfortably.
    min_chars = args.seq_len * args.max_sequences * 8
    print("Reading local WikiText-2 test split ...")
    text = gather_text(min_chars)

    print("Loading SmolLM2-135M ...")
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    token_ids = tokenizer.encode(text)
    seq_len = args.seq_len
    num_windows = min(args.max_sequences, len(token_ids) // seq_len)
    if num_windows == 0:
        print("Not enough tokens to form a single window.")
        return 1
    print(
        f"Evaluating perplexity over {num_windows} window(s) of {seq_len} tokens "
        f"({num_windows * seq_len:,} tokens) ..."
    )

    total_nll = 0.0
    total_tokens = 0
    for w in range(num_windows):
        window = token_ids[w * seq_len : (w + 1) * seq_len]
        inputs = mx.array(window)[None]  # [1, seq_len]
        logits = model(inputs)[0]  # [seq_len, vocab]
        # Predict token t+1 from token t.
        losses = nn.losses.cross_entropy(
            logits[:-1].astype(mx.float32), mx.array(window[1:]), reduction="sum"
        )
        mx.eval(losses)
        total_nll += float(losses)
        total_tokens += seq_len - 1

    mean_nll = total_nll / total_tokens
    perplexity = math.exp(mean_nll)
    print("\n" + "=" * 72)
    print(f"Tokens scored : {total_tokens:,}")
    print(f"Mean NLL      : {mean_nll:.4f} nats/token")
    print(f"Perplexity    : {perplexity:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
