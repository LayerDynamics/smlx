# MiniLM Family

A family of small, efficient sentence transformer models for text embeddings, including all-MiniLM-L6-v2, all-MiniLM-L12-v2, paraphrase-MiniLM, and multi-qa-MiniLM variants.

## Model Variants

### all-MiniLM-L6-v2 (Recommended)

- **Size**: 22.7MB
- **Layers**: 6
- **Dimensions**: 384
- **Best For**: General-purpose embeddings, fastest
- **HuggingFace**: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

### all-MiniLM-L12-v2

- **Size**: ~120MB
- **Layers**: 12
- **Dimensions**: 384
- **Best For**: Better quality, still small
- **HuggingFace**: [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)

### paraphrase-MiniLM-L6-v2

- **Size**: ~22MB
- **Best For**: Paraphrase detection, duplicate finding
- **HuggingFace**: [sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)

### multi-qa-MiniLM-L6-cos-v1

- **Size**: ~22MB
- **Best For**: Question-answer matching, QA systems
- **HuggingFace**: [sentence-transformers/multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1)

## Why MiniLM for SMLX?

MiniLM is the **embedding family** for SMLX:

- Multiple specialized variants for different tasks
- Incredibly small (22MB - 120MB)
- Apache 2.0 license
- Fast inference
- High-quality embeddings
- Perfect for semantic search, clustering, similarity

## Installation

```bash
pip install smlx
```

## Quick Start

```python
from smlx.models.MiniLM import load, encode

# Load specific variant
model, tokenizer = load("all-MiniLM-L6-v2")

# Encode
texts = ["Hello world", "Hi there"]
embeddings = encode(model, tokenizer, texts)
```

## Choosing a Variant

| Variant | Size | Speed | Use Case |
|---------|------|-------|----------|
| all-MiniLM-L6-v2 | 22MB | Fastest | General purpose |
| all-MiniLM-L12-v2 | 120MB | Fast | Higher quality |
| paraphrase-MiniLM | 22MB | Fastest | Duplicate detection |
| multi-qa-MiniLM | 22MB | Fastest | Q&A matching |

## Best Use Cases

MiniLM family excels at:

- ✅ Semantic search
- ✅ Text clustering
- ✅ Similarity comparison
- ✅ Duplicate detection (paraphrase variant)
- ✅ FAQ/QA matching (multi-qa variant)
- ✅ Recommendation systems
- ✅ On-device embeddings

## References

- **HuggingFace**: [sentence-transformers](https://huggingface.co/sentence-transformers)
- **Documentation**: [SBERT](https://www.sbert.net/)

## License

Apache 2.0

---

**Part of the SMLX (smol MLX) project** - Small models optimized for Apple M4 chipsets.
