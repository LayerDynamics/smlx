# all-MiniLM-L6-v2

An ultra-tiny 22MB sentence embedding model that maps text to 384-dimensional vectors for semantic search, clustering, and similarity tasks.

## Model Details

- **Size**: 22.7MB total (tiny!)
- **Type**: Sentence Transformer (BERT-based encoder)
- **Output**: 384-dimensional embeddings
- **Max Sequence**: 256 tokens
- **License**: Apache 2.0
- **HuggingFace**: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## Why all-MiniLM-L6-v2 for SMLX?

all-MiniLM-L6-v2 is the **ultra-lightweight embedding model** for SMLX:

- Incredibly small (22MB!)
- Fast inference
- High-quality embeddings for its size
- Perfect for semantic search and similarity
- Apache 2.0 license
- Most popular sentence embedding model
- Runs anywhere

## Installation

```bash
pip install smlx
```

## Quick Start

### Python API

```python
from smlx.models.all_MiniLM_L6_v2 import load, encode

# Load model
model, tokenizer = load("sentence-transformers/all-MiniLM-L6-v2")

# Encode sentences
sentences = [
    "The cat sits on the mat",
    "A feline rests on a rug",
    "Python is a programming language"
]

embeddings = encode(model, tokenizer, sentences)
# Returns: (3, 384) array

# Compute similarity
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(embeddings)
print(similarities)
```

### Command Line

```bash
# Encode text to embeddings
smlx embed \
  --model all-MiniLM-L6-v2 \
  --text "Hello world" \
  --output embedding.npy
```

## Usage Examples

### Semantic Search

```python
from smlx.models.all_MiniLM_L6_v2 import load, encode
import numpy as np

model, tokenizer = load("sentence-transformers/all-MiniLM-L6-v2")

# Corpus of documents
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Python is a popular programming language",
    "The weather is nice today",
    "Deep learning uses neural networks"
]

# Query
query = "What is AI?"

# Encode
doc_embeddings = encode(model, tokenizer, documents)
query_embedding = encode(model, tokenizer, [query])

# Find most similar
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
best_match_idx = np.argmax(similarities)

print(f"Best match: {documents[best_match_idx]}")
print(f"Similarity: {similarities[best_match_idx]:.3f}")
```

### Text Clustering

```python
from sklearn.cluster import KMeans

# Encode texts
texts = ["text1", "text2", "text3", ...]
embeddings = encode(model, tokenizer, texts)

# Cluster
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(embeddings)

print(f"Cluster assignments: {clusters}")
```

### Duplicate Detection

```python
def find_duplicates(texts, threshold=0.9):
    embeddings = encode(model, tokenizer, texts)
    similarities = cosine_similarity(embeddings)

    duplicates = []
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            if similarities[i][j] > threshold:
                duplicates.append((i, j, similarities[i][j]))

    return duplicates

texts = ["Hello world", "Hi there", "Hello world", "Goodbye"]
dupes = find_duplicates(texts)
print(f"Found {len(dupes)} duplicate pairs")
```

## Performance on M4

| Metric | Value |
|--------|-------|
| Model Size | 22.7MB |
| Memory Usage | ~50MB |
| Encode Speed | ~500 sentences/sec (batch=32) |
| Embedding Dim | 384 |

**Key Strength**: Incredibly fast and tiny while producing quality embeddings.

## Best Use Cases

all-MiniLM-L6-v2 excels at:

- ✅ Semantic search
- ✅ Text similarity comparison
- ✅ Document clustering
- ✅ Duplicate detection
- ✅ FAQ matching
- ✅ Recommendation systems
- ✅ Zero-shot classification (via similarity)
- ✅ On-device embedding generation

## References

- **HuggingFace**: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Paper**: [Sentence-BERT](https://arxiv.org/abs/1908.10084)

## License

Apache 2.0

---

**Part of the SMLX (smol MLX) project** - Small models optimized for Apple M4 chipsets.
