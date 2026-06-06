#!/usr/bin/env python3
"""Debug script for nanoVLM generation to identify why output is empty."""

import logging
import numpy as np
from PIL import Image

import mlx.core as mx
from smlx.models.nanoVLM import load
from smlx.models.nanoVLM.generate import prepare_inputs, sample

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_generation():
    """Debug nanoVLM generation step by step."""

    print("="*80)
    print("LOADING MODEL")
    print("="*80)
    model, processor = load()

    print("\n" + "="*80)
    print("CREATING TEST INPUTS")
    print("="*80)

    # Create simple test image
    test_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    # Use <image> placeholder following mlx-vlm pattern
    prompt = "Describe this <image> in detail:"

    print(f"Prompt: '{prompt}'")
    print(f"Image: {test_image.size}")

    # Prepare inputs
    inputs = prepare_inputs(processor, prompt, test_image)
    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")

    print(f"\nInput IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids.tolist()}")
    print(f"Pixel values shape: {pixel_values.shape if pixel_values is not None else None}")

    print("\n" + "="*80)
    print("GENERATION LOOP")
    print("="*80)

    generated_tokens = []
    max_tokens = 20  # Generate up to 20 tokens for debugging
    temperature = 0.5
    top_p = 1.0
    repetition_penalty = 1.5  # Balanced penalty for coherence without repetition

    for step in range(max_tokens):
        print(f"\n--- Step {step+1} ---")

        # Forward pass
        logits = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
        )

        # Get logits for next token
        next_token_logits = logits[0, -1, :]
        mx.eval(next_token_logits)

        # Print logits statistics
        logits_mean = float(mx.mean(next_token_logits))
        logits_std = float(mx.std(next_token_logits))
        logits_min = float(mx.min(next_token_logits))
        logits_max = float(mx.max(next_token_logits))

        print(f"Logits: mean={logits_mean:.4f}, std={logits_std:.4f}, "
              f"range=[{logits_min:.4f}, {logits_max:.4f}]")

        # Get top-5 tokens
        top5_indices = mx.argpartition(-next_token_logits, kth=5)[:5]
        top5_logits = next_token_logits[top5_indices]
        top5_probs = mx.softmax(next_token_logits)[top5_indices]

        print(f"Top 5 token IDs: {top5_indices.tolist()}")
        print(f"Top 5 logits: {top5_logits.tolist()}")
        print(f"Top 5 probs: {[f'{float(p):.4f}' for p in top5_probs]}")

        # Sample next token with repetition penalty
        next_token = sample(
            next_token_logits,
            temperature,
            top_p,
            previous_tokens=generated_tokens,
            repetition_penalty=repetition_penalty
        )

        print(f"Sampled token: {next_token}")

        # Decode token
        decoded = processor.tokenizer.decode([next_token], skip_special_tokens=False)
        print(f"Decoded: '{decoded}'")

        # Check for EOS
        eos_id = processor.tokenizer.eos_token_id
        print(f"EOS token ID: {eos_id}")

        if next_token == eos_id:
            print("⚠️ Hit EOS token - stopping generation")
            break

        # Add to generated tokens
        generated_tokens.append(next_token)

        # Update input_ids
        next_token_array = mx.array([[next_token]])
        input_ids = mx.concatenate([input_ids, next_token_array], axis=1)
        mx.eval(input_ids)

        # Clear image after first token
        if step == 0:
            pixel_values = None

    print("\n" + "="*80)
    print("GENERATION RESULTS")
    print("="*80)

    print(f"\nGenerated {len(generated_tokens)} tokens")
    print(f"Token IDs: {generated_tokens}")

    if generated_tokens:
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"Generated text: '{generated_text}'")
    else:
        print("⚠️ NO TOKENS GENERATED")

    print("\n" + "="*80)
    print("TOKENIZER INFO")
    print("="*80)

    print(f"Vocab size: {len(processor.tokenizer)}")
    print(f"EOS token ID: {processor.tokenizer.eos_token_id}")
    print(f"BOS token ID: {getattr(processor.tokenizer, 'bos_token_id', 'N/A')}")
    print(f"PAD token ID: {getattr(processor.tokenizer, 'pad_token_id', 'N/A')}")

    # Test encoding/decoding
    test_text = "Hello world"
    encoded = processor.tokenizer.encode(test_text, return_tensors="np")
    decoded = processor.tokenizer.decode(encoded[0], skip_special_tokens=True)
    print(f"\nTest encode/decode: '{test_text}' -> {encoded[0].tolist()} -> '{decoded}'")

if __name__ == "__main__":
    debug_generation()
