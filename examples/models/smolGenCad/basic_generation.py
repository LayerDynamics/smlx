#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
smolGenCad Basic Generation Example.

Demonstrates how to use smolGenCad to generate CAD command sequences from
natural language descriptions.

This example shows:
1. Loading the model and tokenizers
2. Generating CAD from text prompts
3. Validating and auto-fixing sequences
4. Exporting to multiple formats (JSON, Python)
5. Batch generation
"""

from smlx.models.smolGenCad import (
    auto_fix_sequence,
    generate,
    generate_batch,
    load,
    print_model_info,
    sequence_to_dict,
    sequence_to_json,
    sequence_to_python,
    validate_sequence,
)


def example_basic_generation():
    """Basic CAD generation from text."""
    print("\n" + "=" * 70)
    print("Example 1: Basic CAD Generation")
    print("=" * 70)

    # Load model
    print("\nLoading smolGenCad...")
    model, text_tokenizer, cad_tokenizer = load()
    print_model_info(model)

    # Generate CAD from text
    prompt = "Create a cylinder with radius 5cm and height 10cm"
    print(f"\nPrompt: {prompt}")
    print("\nGenerating CAD sequence...")

    sequence = generate(
        model,
        text_tokenizer,
        cad_tokenizer,
        prompt=prompt,
        max_new_tokens=100,
        temperature=0.7,
    )

    print(f"Generated {len(sequence)} CAD commands:")
    for i, (command, params) in enumerate(sequence, 1):
        print(f"  {i}. {command.name}: {params}")


def example_validation():
    """Demonstrate validation and auto-fixing."""
    print("\n" + "=" * 70)
    print("Example 2: Validation and Auto-Fixing")
    print("=" * 70)

    from smlx.models.smolGenCad.commands import CADCommandType

    # Create a potentially invalid sequence
    sequence = [
        (CADCommandType.CIRCLE, {"cx": 0, "cy": 0, "r": 50}),
        (CADCommandType.EXTRUDE, {"distance": 100}),
    ]

    print("\nOriginal sequence (missing SKETCH_START/END):")
    for cmd, params in sequence:
        print(f"  {cmd.name}: {params}")

    # Validate
    is_valid, errors = validate_sequence(sequence)
    print(f"\nValidation: {'✓ Valid' if is_valid else '✗ Invalid'}")
    if not is_valid:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")

    # Auto-fix
    print("\nAuto-fixing sequence...")
    fixed_sequence = auto_fix_sequence(sequence)

    print("\nFixed sequence:")
    for cmd, params in fixed_sequence:
        print(f"  {cmd.name}: {params}")

    # Validate fixed sequence
    is_valid, errors = validate_sequence(fixed_sequence)
    print(f"\nFixed validation: {'✓ Valid' if is_valid else '✗ Invalid'}")


def example_export_formats():
    """Demonstrate exporting to different formats."""
    print("\n" + "=" * 70)
    print("Example 3: Export Formats")
    print("=" * 70)

    model, text_tokenizer, cad_tokenizer = load()

    # Generate a simple CAD sequence
    prompt = "Create a rectangular box 10x20x5"
    print(f"\nPrompt: {prompt}")

    sequence = generate(
        model, text_tokenizer, cad_tokenizer, prompt, temperature=0.7
    )

    # Export to dictionary
    print("\n1. Dictionary Format:")
    dict_format = sequence_to_dict(sequence)
    print(dict_format[:2])  # Show first 2 commands
    print("...")

    # Export to JSON
    print("\n2. JSON Format:")
    json_format = sequence_to_json(sequence)
    print(json_format[:200] + "...")

    # Export to Python (CadQuery)
    print("\n3. Python (CadQuery) Format:")
    python_code = sequence_to_python(sequence)
    print(python_code)


def example_batch_generation():
    """Demonstrate batch generation."""
    print("\n" + "=" * 70)
    print("Example 4: Batch Generation")
    print("=" * 70)

    model, text_tokenizer, cad_tokenizer = load()

    # Multiple prompts
    prompts = [
        "Create a cylinder with radius 10mm and height 20mm",
        "Create a cube with side 15mm and rounded corners",
        "Create a hollow sphere with diameter 30mm",
    ]

    print("\nGenerating CAD for multiple prompts...")
    print(f"Prompts: {len(prompts)}")

    sequences = generate_batch(
        model, text_tokenizer, cad_tokenizer, prompts, temperature=0.7
    )

    print(f"\nGenerated {len(sequences)} CAD sequences:\n")
    for i, (prompt, sequence) in enumerate(zip(prompts, sequences), 1):
        print(f"{i}. Prompt: {prompt}")
        print(f"   Commands: {len(sequence)}")
        print(f"   First 3 operations:")
        for cmd, params in sequence[:3]:
            print(f"     - {cmd.name}: {params}")
        print()


def example_custom_parameters():
    """Demonstrate generation with custom parameters."""
    print("\n" + "=" * 70)
    print("Example 5: Custom Generation Parameters")
    print("=" * 70)

    model, text_tokenizer, cad_tokenizer = load()

    prompt = "Create a complex mechanical part with multiple features"

    # Different temperature settings
    print("\nComparing different temperatures:\n")

    for temp in [0.5, 0.8, 1.0]:
        print(f"Temperature: {temp}")
        sequence = generate(
            model,
            text_tokenizer,
            cad_tokenizer,
            prompt,
            temperature=temp,
            max_new_tokens=50,
        )
        print(f"  Generated {len(sequence)} commands")
        print(f"  Operations: {[cmd.name for cmd, _ in sequence[:5]]}")
        print()


def example_sequence_analysis():
    """Analyze generated CAD sequences."""
    print("\n" + "=" * 70)
    print("Example 6: Sequence Analysis")
    print("=" * 70)

    model, text_tokenizer, cad_tokenizer = load()

    prompt = "Create a cylindrical part with filleted edges"
    print(f"\nPrompt: {prompt}\n")

    sequence = generate(model, text_tokenizer, cad_tokenizer, prompt)

    # Analyze sequence composition
    from collections import Counter

    command_counts = Counter(cmd.name for cmd, _ in sequence)

    print("Command composition:")
    for cmd_name, count in command_counts.most_common():
        print(f"  {cmd_name}: {count}")

    # Analyze by category
    categories = {"control": 0, "sketch": 0, "feature": 0, "refinement": 0}
    for cmd, _ in sequence:
        categories[cmd.category] += 1

    print("\nBy category:")
    for category, count in categories.items():
        print(f"  {category}: {count}")

    # Check sequence length
    print(f"\nTotal operations: {len(sequence)}")
    print(
        f"Max allowed: {model.config.vocabulary.max_sequence_length}"
    )


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("smolGenCad: Basic Generation Examples")
    print("World's Smallest CAD Generation Model (158M parameters)")
    print("=" * 70)

    try:
        # Run examples
        example_basic_generation()
        example_validation()
        example_export_formats()
        example_batch_generation()
        example_custom_parameters()
        example_sequence_analysis()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Try your own prompts with generate()")
        print("2. Explore different generation parameters")
        print("3. Export sequences to Python and execute with CadQuery")
        print("4. Train the model on CAD datasets for better results")
        print("\nSee README.md for more information.")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
