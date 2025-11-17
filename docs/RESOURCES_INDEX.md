# Resources Documentation Index

This directory contains comprehensive documentation analyzing reusable patterns from the `resources/` directory for the SMLX project.

## Documentation Files

### 1. RESOURCES_QUICK_START.md (8 KB, 1,039 words)

**Start here for quick implementation**

- TL;DR - 3 most important files to study
- 6-step implementation checklist for new models
- Copy-paste code blocks from patterns
- Common pitfalls and solutions
- Integration points with SMLX utilities
- FAQ section with quick answers

Best for: Getting started quickly, implementation checklists, common questions

### 2. RESOURCES_REFERENCE_MAP.md (15 KB, 1,541 words)

**Use for copy-paste code patterns**

- Quick reference table of file locations
- Exact line numbers for each pattern
- 8 complete code patterns ready to copy
- File dependencies and import structure
- Testing patterns for each model type
- Step-by-step copy-paste checklist

Best for: Finding exact code to copy, locating patterns, testing approaches

### 3. RESOURCES_PATTERNS.md (24 KB, 2,564 words)

**Deep dive into architectural patterns**

- 17 major sections covering all aspects
- Detailed explanations of each pattern
- Design rationale and best practices
- Code examples with full context
- Anti-patterns to avoid
- Model type-specific guidance (text, vision, audio, multimodal)

Best for: Understanding design choices, learning best practices, comprehensive reference

## How to Use These Documents

### If you want to implement a new model quickly

1. Read: RESOURCES_QUICK_START.md (5 min)
2. Reference: RESOURCES_REFERENCE_MAP.md (copy code patterns)
3. Implement: Follow the checklist in QUICK_START
4. Test: Use testing patterns from REFERENCE_MAP

### If you want to understand the architecture

1. Read: RESOURCES_PATTERNS.md (20 min)
2. Study: Files from "Recommended Study Order" section
3. Reference: RESOURCES_REFERENCE_MAP.md for exact line numbers
4. Implement: With deep understanding of why things work

### If you want to find specific patterns

1. Use: RESOURCES_REFERENCE_MAP.md table of contents
2. Search: "Pattern N: [Pattern Name]" to find code
3. Note: Exact line numbers in resource files
4. Copy: Code patterns directly, adapt names

## Document Organization

### RESOURCES_QUICK_START.md Sections

1. TL;DR - 3 most important files
2. Implementation checklist (6 steps)
3. Copy-paste code blocks
4. Where each component comes from
5. Quickest path (Gemma example)
6. Testing procedures
7. Common pitfalls
8. File structure template
9. SMLX utilities integration
10. Key principles
11. FAQ

### RESOURCES_REFERENCE_MAP.md Sections

1. Quick reference tables
2. Most important files (prioritized)
3. Exact code patterns (8 patterns)
4. File dependencies
5. Testing patterns
6. Copy-paste checklist

### RESOURCES_PATTERNS.md Sections

1. Executive summary
2. Directory structure overview
3. Core architecture patterns (models, cache, generation, sampling, tokenizer, loading)
4. Quantization patterns (GPTQ, AWQ, LoRA)
5. Vision-language patterns
6. Audio patterns
7. Utilities and helpers
8. Server/API patterns
9. Patterns by model type
10. Conversion utilities
11. Training patterns
12. Serialization formats
13. Key reusable utilities table
14. Anti-patterns to avoid
15. Recommended implementation approach
16. Summary of most important patterns

## Key Resources Referenced

All patterns are found in `/Users/ryanoboyle/smlx/resources/`:

| Source | Type | Key File | Size |
|--------|------|----------|------|
| mlx-lm | Language Models (100+) | mlx_lm/models/gemma.py | 180 lines |
| mlx-vlm | Vision-Language (36+) | mlx_vlm/models/smolvlm/ | 64 lines |
| mlx-examples | Examples | examples/llms/ | Varies |
| lightning-whisper-mlx | Audio | whisper.py | 400 lines |
| mflux | Diffusion | N/A | 5000+ lines |

## Critical Patterns Summary

### 7 Most Important Patterns (All Models)

1. **BaseModelArgs** - Configuration pattern (copy exactly)
2. **Attention** - Multi-head attention with RoPE (adapt)
3. **MLP** - Feed-forward network (usually copy)
4. **TransformerBlock** - Composition of Attention + MLP (copy)
5. **KV Cache** - Efficient caching for generation (copy/use)
6. **Sampling** - Composable sampling pipeline (use directly)
7. **Generation** - Token-by-token generation with cache (adapt)

### 3 Most Important Files

1. `/resources/mlx-lm/mlx_lm/models/base.py` - Foundation
2. `/resources/mlx-lm/mlx_lm/models/gemma.py` - Clean example
3. `/resources/mlx-lm/mlx_lm/models/cache.py` - KV cache

## Implementation Workflow

```
START HERE
    |
    v
Read RESOURCES_QUICK_START.md (5 minutes)
    |
    v
Decide: Do I need deep understanding?
    |
    +--- YES --> Read RESOURCES_PATTERNS.md (30 minutes)
    |
    +--- NO --> Continue
    |
    v
Reference RESOURCES_REFERENCE_MAP.md
    |
    v
Find pattern (e.g., "Pattern 1: BaseModelArgs")
    |
    v
Copy exact code with line numbers
    |
    v
Follow 6-step checklist from QUICK_START
    |
    v
Test using patterns from REFERENCE_MAP
    |
    v
DONE!
```

## When to Reference Each Document

**RESOURCES_QUICK_START.md:**

- Need quick overview (5 min read)
- Want step-by-step checklist
- Looking for common pitfalls
- Need FAQ answers
- First time reading docs

**RESOURCES_REFERENCE_MAP.md:**

- Have specific pattern to find
- Need exact code to copy
- Looking for line numbers
- Want testing examples
- File dependency questions

**RESOURCES_PATTERNS.md:**

- Want comprehensive understanding
- Learning best practices
- Implementing complex model types
- Understanding design decisions
- Need complete reference

## File Names in Patterns

When patterns refer to files, they use paths like:

- `/mlx-lm/mlx_lm/models/base.py`
- `/mlx-vlm/mlx_vlm/models/smolvlm/`
- `/mlx-examples/llms/mixtral/`

These are relative to:

```
/Users/ryanoboyle/smlx/resources/
```

Complete paths:

```
/Users/ryanoboyle/smlx/resources/mlx-lm/mlx_lm/models/base.py
/Users/ryanoboyle/smlx/resources/mlx-vlm/mlx_vlm/models/smolvlm/
```

## Key Principles

1. **Copy, don't import** - Adapt code from resources into SMLX
2. **Reuse utilities** - Use already-implemented functions (cache, sampling, generation)
3. **Quantization first** - All models must support 4-bit and 8-bit
4. **Small models** - Keep parameter count under 1B
5. **Streaming output** - Support incremental text generation
6. **Efficient caching** - Use KV cache for efficiency
7. **Clean interfaces** - Export load(), generate(), chat()

## Quick Links

- **SMLX Project**: `/Users/ryanoboyle/smlx/`
- **Resources**: `/Users/ryanoboyle/smlx/resources/`
- **Models**: `/Users/ryanoboyle/smlx/smlx/models/`
- **Examples**: `/Users/ryanoboyle/smlx/examples/`
- **Tests**: `/Users/ryanoboyle/smlx/tests/`

## Support

For questions about:

- **Quick start**: See RESOURCES_QUICK_START.md FAQ section
- **Code patterns**: See RESOURCES_REFERENCE_MAP.md "Exact Code Pattern Examples"
- **Design**: See RESOURCES_PATTERNS.md relevant section
- **Implementation**: Follow checklist in RESOURCES_QUICK_START.md

## Document Versions

Created: November 12, 2025
Format: Markdown
Total: 47 KB, 5,144 words across 3 documents
Tools: Claude Code, MLX framework

---

Start with **RESOURCES_QUICK_START.md** and follow from there!
