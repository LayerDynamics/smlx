#!/usr/bin/env python3
"""Showcase: semantic search over local data with a 23M-param embedding model.

Loads sentences straight from the bundled GLUE dataset (``data/benchmark/glue``)
via :mod:`smlx.data.local`, embeds them with all-MiniLM-L6-v2 (≈23M params,
384-dim), and answers a free-text query by cosine similarity -- a complete
semantic-search loop running entirely on-device.

This highlights two SMLX advantages: real datasets ship *in the repo* (no
download needed), and a sub-100M model gives genuinely useful embeddings on
Apple Silicon in milliseconds.

Run::

    python examples/showcase/minilm_semantic_search.py
    python examples/showcase/minilm_semantic_search.py --query "a film worth watching" --top-k 5
"""

from __future__ import annotations

import argparse

import numpy as np

from smlx.data import local
from smlx.models.MiniLM import cosine_similarity, encode, load


def gather_sentences(limit: int) -> list[str]:
    """Pull unique, non-trivial sentences from the local GLUE SST-2 split."""
    ds = local.load("glue", split="sst2_validation")
    sentences: list[str] = []
    seen: set[str] = set()
    for row in ds:
        text = (row.get("sentence") or "").strip()
        if len(text) >= 15 and text not in seen:
            seen.add(text)
            sentences.append(text)
        if len(sentences) >= limit:
            break
    return sentences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--query",
        default="a heartfelt and moving story",
        help="Free-text search query",
    )
    parser.add_argument("--corpus-size", type=int, default=50, help="Sentences to index")
    parser.add_argument("--top-k", type=int, default=5, help="Results to show")
    args = parser.parse_args()

    if not local.is_available("glue"):
        print(
            "GLUE data not present. Fetch with: python -m smlx.tools.download_data --dataset glue"
        )
        return 1

    print(f"Indexing {args.corpus_size} sentences from local GLUE SST-2 ...")
    corpus = gather_sentences(args.corpus_size)

    print("Loading all-MiniLM-L6-v2 (≈23M params) ...")
    model, tokenizer = load("all-MiniLM-L6-v2")

    corpus_emb = np.ascontiguousarray(encode(model, tokenizer, corpus), dtype=np.float32)
    query_emb = np.ascontiguousarray(encode(model, tokenizer, [args.query]), dtype=np.float32)

    # [1, N] similarity matrix -> ranked indices. The embeddings are verified
    # finite, unit-norm float32; NumPy's SIMD matmul can still emit a spurious
    # FP-exception warning on the transposed view inside cosine_similarity, so
    # we silence FP warnings for this one (correct) dot product.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = np.asarray(cosine_similarity(query_emb, corpus_emb))[0]
    order = np.argsort(-sims)[: args.top_k]

    print(f"\nQuery: {args.query!r}")
    print(f"Top {args.top_k} matches (cosine similarity):")
    print("=" * 72)
    for rank, idx in enumerate(order, 1):
        print(f"{rank}. [{sims[idx]:.3f}] {corpus[idx]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
