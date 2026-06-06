#!/usr/bin/env python3
"""Showcase: vision-language Q&A over local images with a 256M-param VLM.

Runs SmolVLM-256M-Instruct on the bundled COCO8 images
(``data/datasets/coco8``) loaded through :mod:`smlx.data.local`, captioning each
image and answering a follow-up question -- a full multimodal pipeline on a
quarter-billion-parameter model that fits comfortably on an M-series Mac.

Run::

    python examples/showcase/smolvlm_coco8_vqa.py
    python examples/showcase/smolvlm_coco8_vqa.py --num-images 2 --question "How many objects are there?"
"""

from __future__ import annotations

import argparse

from PIL import Image

from smlx.data import local
from smlx.models.SmolVLM_256M import generate, load


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--num-images", type=int, default=3, help="Images to caption")
    parser.add_argument(
        "--question",
        default="What is the main object in this image?",
        help="Follow-up question asked about each image",
    )
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens per answer")
    parser.add_argument("--split", default="train", help="COCO8 split (train/val)")
    args = parser.parse_args()

    if not local.is_available("coco8"):
        print("COCO8 not present. Fetch with: python -m smlx.tools.download_data --dataset coco8")
        return 1

    tree = local.load("coco8", split=args.split)
    images = tree.images[: args.num_images]
    print(f"Loaded {len(images)} local COCO8 image(s) from split '{tree.split}'.")

    print("Loading SmolVLM-256M-Instruct ...")
    model, processor = load("HuggingFaceTB/SmolVLM-256M-Instruct")

    for idx, image_path in enumerate(images, 1):
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        rel = image_path.relative_to(tree.root)
        print("\n" + "=" * 72)
        print(f"[{idx}/{len(images)}] {rel}  ({image.size[0]}x{image.size[1]})")
        print("-" * 72)

        caption = generate(
            model,
            processor,
            prompt="Describe this image in one sentence.",
            image=image,
            max_tokens=args.max_tokens,
            temperature=0.0,
        )
        print(f"Caption : {caption.strip()}")

        answer = generate(
            model,
            processor,
            prompt=args.question,
            image=image,
            max_tokens=args.max_tokens,
            temperature=0.0,
        )
        print(f"Q       : {args.question}")
        print(f"A       : {answer.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
