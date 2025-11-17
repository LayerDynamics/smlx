# Contributing to SMLX

Thank you for your interest in contributing to SMLX! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Contributing Models](#contributing-models)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/yourusername/smlx.git
cd smlx

# Add upstream remote
git remote add upstream https://github.com/originalowner/smlx.git
```

### Development Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate smlx

# Or use pip
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pre-commit install
```

### Verify Setup

```bash
# Run tests
python -m pytest

# Check code style
black --check .
ruff check .

# Type check
mypy smlx/
```

## Development Workflow

### 1. Create a Branch

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clear, concise code
- Follow existing patterns and conventions
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
python -m pytest

# Run specific tests
python -m pytest tests/your_test.py -v

# Check coverage
python -m pytest --cov=smlx --cov-report=html
```

### 4. Format and Lint

```bash
# Format code
black .

# Lint
ruff check .
ruff check --fix .  # Auto-fix issues

# Type check
mypy smlx/
```

### 5. Commit Changes

Use clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: description of feature"
```

**Commit message format:**

```text
<type>: <subject>

<body>

<footer>
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Example:**

```text
feat: add Donut-base document model

Implement Donut-base for document understanding with:
- Model architecture in model.py
- Loading from HuggingFace Hub
- Document parsing interface
- Examples and tests

Closes #123
```

### 6. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

## Contributing Models

**IMPORTANT: SMLX only accepts "smol" models (< 1B parameters).**

### Model Requirements

Before contributing a model, ensure it meets these requirements:

1. **Size:** < 1B parameters (< 500M preferred)
2. **License:** Permissive license (MIT, Apache 2.0, BSD)
3. **Quality:** Proven performance on benchmarks
4. **MLX Compatible:** Works with MLX framework
5. **Documentation:** Clear usage examples and documentation

### Model Implementation Checklist

- [ ] Model has < 1B parameters
- [ ] Create model directory: `smlx/models/YourModel/`
- [ ] Implement required modules:
  - [ ] `config.py` - Model configuration
  - [ ] `model.py` - Model architecture (MLX-based)
  - [ ] `loader.py` - Loading from HuggingFace Hub
  - [ ] `generate.py` or `classify.py` - Inference interface
  - [ ] `__init__.py` - Public API exports
- [ ] Add example: `examples/models/your_model/`
- [ ] Add integration test: `tests/integration/test_your_model.py`
- [ ] Quantization support (4-bit/8-bit)
- [ ] Documentation:
  - [ ] Docstrings in all modules
  - [ ] Usage examples in README
  - [ ] Model card (architecture, training, limitations)
- [ ] Performance benchmarks

### Model Structure Template

See [docs/ModelImplementations.md](docs/ModelImplementations.md) for detailed template.

**Minimal example:**

```python
# smlx/models/YourModel/model.py
import mlx.core as mx
import mlx.nn as nn

class YourModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Define layers...

    def __call__(self, x):
        # Forward pass...
        return x

# smlx/models/YourModel/loader.py
from pathlib import Path
from huggingface_hub import snapshot_download
from .model import YourModel
from .config import DEFAULT_CONFIG

def load(model_name: str = "default"):
    """Load model from HuggingFace Hub."""
    # Download model
    path = snapshot_download(repo_id=model_name)

    # Load config and weights
    config = DEFAULT_CONFIG
    model = YourModel(config)

    # Load weights...
    model.eval()
    return model
```

## Code Style

### Python Style

We follow PEP 8 with some modifications:

- **Line length:** 100 characters (configured in pyproject.toml for Black and Ruff)
- **Quotes:** Double quotes for strings
- **Imports:** Organized with `isort`
- **Type hints:** Use type hints for function signatures

### Formatting Tools

```bash
# Auto-format with Black
black .

# Sort imports
isort .

# Lint with Ruff
ruff check .
ruff check --fix .
```

### Code Quality

- Write clear, self-documenting code
- Use descriptive variable names
- Add docstrings to all public functions/classes
- Avoid premature optimization
- Keep functions focused and small
- Use type hints

### Docstring Format

Use Google-style docstrings:

```python
def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
) -> str:
    """
    Generate text from a prompt.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input text prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)

    Returns:
        Generated text string

    Raises:
        ValueError: If max_tokens < 1

    Example:
        >>> model, tokenizer = load("SmolLM2-135M")
        >>> output = generate(model, tokenizer, "Hello", max_tokens=50)
        >>> print(output)
    """
    # Implementation...
```

## Testing

### Test Organization

```text
tests/
├── unit/              # Fast, isolated tests
├── integration/       # Integration tests (may download models)
├── benchmark/         # Performance benchmarks
└── conftest.py        # Pytest configuration and fixtures
```

### Writing Tests

Use pytest with custom markers:

```python
import pytest
from smlx.models.YourModel import load, generate

@pytest.mark.unit
def test_model_config():
    """Test model configuration."""
    from smlx.models.YourModel.config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG.hidden_size == 512

@pytest.mark.integration
@pytest.mark.requires_model
def test_model_loading():
    """Test model loading from HuggingFace Hub."""
    model = load("your-org/your-model")
    assert model is not None

@pytest.mark.slow
@pytest.mark.gpu
def test_model_inference():
    """Test model inference."""
    model = load("your-org/your-model")
    output = generate(model, "Hello", max_tokens=10)
    assert len(output) > 0
```

### Running Tests

```bash
# All tests
python -m pytest

# Unit tests only (fast)
python -m pytest -m unit

# Skip slow tests
python -m pytest -m "not slow"

# Specific file
python -m pytest tests/unit/test_config.py -v

# With coverage
python -m pytest --cov=smlx --cov-report=html
```

### Test Coverage

Aim for:

- **Unit tests:** > 80% coverage
- **Integration tests:** Core functionality
- **Edge cases:** Error handling, boundary conditions

## Documentation

### Documentation Requirements

All contributions should include appropriate documentation:

1. **Docstrings:** All public functions, classes, methods
2. **README updates:** If adding new features
3. **Examples:** Working code examples
4. **Type hints:** Function signatures
5. **Comments:** For complex logic only

### Documentation Structure

```text
docs/
├── ModelImplementations.md   # Model implementation guide
├── Server.md                  # API server documentation
├── Agents.md                  # Agent system guide
├── Tools.md                   # CLI tools reference
├── Quant.md                   # Quantization techniques
└── Eval.md                    # Evaluation framework
```

### Writing Documentation

- Use clear, concise language
- Provide working code examples
- Include expected output
- Document limitations and gotchas
- Link to related documentation

### Example Documentation

```markdown
## YourModel

Brief description of the model.

### Installation

\`\`\`bash
pip install -e ".[your_feature]"
\`\`\`

### Quick Start

\`\`\`python
from smlx.models.YourModel import load, generate

# Load model
model = load("your-org/your-model")

# Generate
output = generate(model, "Hello", max_tokens=50)
print(output)
\`\`\`

### API Reference

See [YourModel API](docs/api/yourmodel.md) for details.

### Examples

- [Basic usage](examples/yourmodel/basic.py)
- [Advanced features](examples/yourmodel/advanced.py)

### Performance

| Metric | Value |
|--------|-------|
| Size   | 250M  |
| Speed  | 150 tok/s |
```

## Pull Request Process

### Before Submitting

1. **Update your branch:**

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run full test suite:**

   ```bash
   python -m pytest
   ```

3. **Format and lint:**

   ```bash
   black .
   ruff check .
   mypy smlx/
   ```

4. **Update documentation:**
   - README.md if adding features
   - Relevant docs/ files
   - Docstrings

### PR Title Format

Use clear, descriptive titles:

```text
feat: add Donut-base document model
fix: resolve memory leak in KV cache
docs: update quantization guide
test: add integration tests for SmolVLM
```

### PR Description Template

```markdown
## Description

Brief description of changes.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No new warnings introduced

## Testing

Describe testing performed:
- Unit tests: ...
- Integration tests: ...
- Manual testing: ...

## Additional Context

Any additional information, screenshots, benchmarks, etc.
```

### Review Process

1. **Automated checks:** CI/CD runs tests, linting, type checking
2. **Code review:** Maintainers review code
3. **Feedback:** Address reviewer comments
4. **Approval:** At least one maintainer approval required
5. **Merge:** Squash and merge to main

### After Merge

- Delete your feature branch
- Update your fork:

  ```bash
  git checkout main
  git pull upstream main
  git push origin main
  ```

## Areas for Contribution

### High Priority

- **New models:** Small models (< 1B params) in categories:
  - Language models
  - Vision-language models
  - Audio models
  - Document models
  - Embedding models

- **Performance optimization:**
  - Faster inference
  - Lower memory usage
  - Better quantization

- **Documentation:**
  - More examples
  - Tutorials
  - API documentation

### Medium Priority

- **Testing:**
  - Increase test coverage
  - Add edge case tests
  - Performance regression tests

- **Tools:**
  - Better benchmarking
  - Model conversion utilities
  - Deployment tools

- **Evaluation:**
  - More benchmark datasets
  - Better metrics
  - Automated evaluation

### Low Priority

- **Infrastructure:**
  - CI/CD improvements
  - Docker optimization
  - Kubernetes configs

## Questions?

- **GitHub Issues:** <https://github.com/yourusername/smlx/issues>
- **Discussions:** <https://github.com/yourusername/smlx/discussions>
- **Email:** <maintainer@example.com>

## License

By contributing to SMLX, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to SMLX!** 🚀
