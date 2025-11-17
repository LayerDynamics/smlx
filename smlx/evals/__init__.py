"""
SMLX Evaluation Benchmarks.

This module provides comprehensive evaluation benchmarks for vision-language models
running on Apple Silicon using the MLX framework. All benchmarks are optimized for
small models and include detailed metrics, category breakdowns, and CSV/JSON outputs.

Available Benchmarks:
- MathVista: Math-Vision-Language reasoning (6,141 examples)
- MMMU: Massive Multi-discipline Multimodal Understanding (30 subjects)
- MMStar: Elite vision-indispensable benchmark (1,500 samples, 6 categories)
- OCRBench: OCR capabilities evaluation (1,000 samples)

Example usage:
    # Run benchmarks via command line (recommended)
    python -m smlx.evals.math_vista --model <model_id>
    python -m smlx.evals.mmmu --subset Math --max-samples 10
    python -m smlx.evals.mmstar --model <model_id>
    python -m smlx.evals.ocrbench --model <model_id>

    # Or programmatically use utility functions
    from smlx.evals import inference, load_model

    model, processor = load_model("mlx-community/SmolVLM-256M-Instruct")
    response = inference(model, processor, "What is in this image?", image)
"""

# Only export utility functions for programmatic use
from smlx.evals.utils import inference, load_model

__all__ = [
    "inference",
    "load_model",
]
