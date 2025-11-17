# SMLX Quick Start Guide

Get up and running with SMLX in 5 minutes.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python >= 3.9, < 3.13
- Xcode Command Line Tools

## Installation

### Option 1: Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/smlx.git
cd smlx

# Create environment
conda env create -f environment.yml
conda activate smlx
```

### Option 2: pip

```bash
# Clone repository
git clone https://github.com/yourusername/smlx.git
cd smlx

# Install
pip install -e ".[all]"
```

## Your First Model

### 1. Basic Text Generation

Create `hello_smlx.py`:

```python
from smlx.models.SmolLM2_135M import load, generate

# Load model (downloads automatically from HuggingFace)
print("Loading model...")
model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

# Generate text
prompt = "The future of AI is"
print(f"\nPrompt: {prompt}")
print("Response:", generate(model, tokenizer, prompt, max_tokens=50))
```

Run it:

```bash
python hello_smlx.py
```

**Output:**

```text
Loading model...
Downloading model from HuggingFace Hub...

Prompt: The future of AI is
Response: The future of AI is exciting and full of possibilities...
```

### 2. Streaming Chat

Create `chat_example.py`:

```python
from smlx.models.SmolLM2_135M import load, chat

# Load model
model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

# Chat conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain machine learning in one sentence."}
]

# Stream response
print("Assistant: ", end="", flush=True)
for chunk in chat(model, tokenizer, messages, stream=True):
    print(chunk, end="", flush=True)
print()  # New line
```

Run it:

```bash
python chat_example.py
```

### 3. Vision-Language Model

Create `vision_example.py`:

```python
from smlx.models.SmolVLM_256M import load, generate
from PIL import Image
import requests
from io import BytesIO

# Load model
print("Loading vision-language model...")
model, processor = load("HuggingFaceTB/SmolVLM-256M-Instruct")

# Download example image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

# Ask question about image
prompt = "What color is the car in this image?"
answer = generate(model, processor, prompt, image)
print(f"\nQuestion: {prompt}")
print(f"Answer: {answer}")
```

Run it:

```bash
python vision_example.py
```

### 4. Audio Transcription

Create `audio_example.py`:

```python
from smlx.models.Whisper_tiny import load, transcribe
import requests

# Load model
print("Loading audio model...")
model, processor = load()

# Download example audio (or use your own)
# For this example, we'll use a local file
audio_path = "your_audio.wav"

# Transcribe
result = transcribe(model, processor, audio_path)
print(f"\nTranscription: {result['text']}")
```

### 5. Document OCR

Create `ocr_example.py`:

```python
from smlx.models.TrOCR_small import load, recognize
from PIL import Image, ImageDraw, ImageFont

# Create a simple text image
def create_test_image(text="Hello SMLX!"):
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
    except:
        font = ImageFont.load_default()
    draw.text((10, 30), text, fill='black', font=font)
    return img

# Load model (printed text variant)
print("Loading OCR model...")
model, processor = load("printed")

# Create and recognize text
test_text = "Hello SMLX!"
image = create_test_image(test_text)
recognized = recognize(model, processor, image)

print(f"\nOriginal: {test_text}")
print(f"Recognized: {recognized}")
```

Run it:

```bash
python ocr_example.py
```

## Common Patterns

### Pattern 1: Custom Generation Parameters

```python
from smlx.models.SmolLM2_135M import load, generate

model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

output = generate(
    model,
    tokenizer,
    prompt="Write a haiku about coding:",
    max_tokens=100,
    temperature=0.8,      # Higher = more creative
    top_p=0.95,           # Nucleus sampling
    repetition_penalty=1.1  # Reduce repetition
)
print(output)
```

### Pattern 2: Batch Processing

```python
from smlx.models.SmolLM2_135M import load, generate

model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

prompts = [
    "What is Python?",
    "What is MLX?",
    "What is SMLX?"
]

for prompt in prompts:
    response = generate(model, tokenizer, prompt, max_tokens=50)
    print(f"Q: {prompt}")
    print(f"A: {response}\n")
```

### Pattern 3: Model Quantization

```python
from smlx.models.SmolLM2_135M import load, generate
from smlx.quant import quantize_model

# Load model
model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

# Quantize to 4-bit (reduces memory by ~75%)
print("Quantizing model to 4-bit...")
quantized_model = quantize_model(model, bits=4, group_size=64)

# Use quantized model (same API)
output = generate(quantized_model, tokenizer, "Hello", max_tokens=50)
print(output)
```

### Pattern 4: Error Handling

```python
from smlx.models.SmolLM2_135M import load, generate

try:
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
    output = generate(model, tokenizer, "Hello", max_tokens=50)
    print(output)
except Exception as e:
    print(f"Error: {e}")
    print("Make sure you have internet connection for first-time model download.")
```

## Using the CLI Tools

### Download Models

```bash
# Download specific model
python -m smlx.tools.download_data --model mlx-community/SmolLM2-135M-Instruct

# Download all models
python -m smlx.tools.download_data --models

# Check download status
python -m smlx.tools.download_data --list
```

### Convert Models

```bash
# Convert HuggingFace model to MLX format
python -m smlx.tools.convert2mlx \
  --hf-path gpt2 \
  --output-path ./models/gpt2-mlx

# Convert with quantization
python -m smlx.tools.convert2mlx \
  --hf-path gpt2 \
  --output-path ./models/gpt2-4bit \
  --quantize \
  --bits 4 \
  --group-size 64
```

### Run Benchmarks

```bash
# Benchmark your system
python -m smlx.bench.run

# Benchmark specific model
python -m smlx.bench.run --model SmolLM2-135M
```

## Running the Server

### Start the API Server

```bash
# Start server
python -m smlx.server.app --host 0.0.0.0 --port 8000
```

### Test with curl

```bash
# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SmolLM2-135M-Instruct",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'

# Streaming response
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SmolLM2-135M-Instruct",
    "messages": [
      {"role": "user", "content": "Write a poem"}
    ],
    "stream": true
  }'
```

### Test with Python

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "SmolLM2-135M-Instruct",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ]
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

## Using Agents

### Basic ReAct Agent

Create `agent_example.py`:

```python
from smlx.agents import ReActAgent
from smlx.agents.tools import ToolRegistry, calculator, get_time
from smlx.models.SmolLM2_135M import load

# Load model
model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

# Create tool registry
registry = ToolRegistry()
registry.register(calculator)
registry.register(get_time)

# Create agent
agent = ReActAgent(model, tokenizer, registry, max_iterations=5)

# Run task
task = "What is 25 * 17, and what time is it?"
response = agent.run(task)

print(f"Task: {task}")
print(f"Result: {response.content}")
print(f"Reasoning: {response.reasoning}")
```

Run it:

```bash
python agent_example.py
```

### Chain-of-Thought Agent

```python
from smlx.agents import CoTAgent
from smlx.models.SmolLM2_135M import load

# Load model
model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

# Create agent
agent = CoTAgent(model, tokenizer, zero_shot=True)

# Solve problem with step-by-step reasoning
task = "If a train travels 60 mph for 2.5 hours, how far does it go?"
response = agent.run(task)

print(f"Task: {task}")
print(f"Answer: {response.content}")
print(f"\nReasoning:\n{response.reasoning}")
```

## Exploring Examples

The `examples/` directory contains working examples for all features:

```bash
# Language models
python examples/models/smollm2_135m/smollm2_135m_example.py

# Vision-language models
python examples/models/smolvlm_256m/smolvlm_example.py

# Audio models
python examples/whisper_tiny/basic_transcription.py

# OCR
python examples/models/trocr_small/trocr_example.py

# Quantization
python examples/quant/gptq_example.py
python examples/quant/lora_example.py

# Agents
python examples/agents/react_agent_example.py
python examples/agents/cot_agent_example.py

# Evaluation
python examples/eval/mmmu_eval.py
```

## Running Tests

Verify your installation:

```bash
# Run all tests
python -m pytest

# Run quick tests only
python -m pytest -m "not slow"

# Run specific test
python -m pytest tests/integration/test_smollm2_generation.py -v

# Run with verbose output
python -m pytest -v
```

## Troubleshooting

### Issue: Model download fails

**Solution:**

```bash
# Check internet connection
ping huggingface.co

# Try manual download
python -m smlx.tools.download_data --model mlx-community/SmolLM2-135M-Instruct --force
```

### Issue: Out of memory errors

**Solution:**

```python
# Use quantization to reduce memory usage
from smlx.quant import quantize_model

model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
model = quantize_model(model, bits=4, group_size=64)  # 75% memory reduction
```

### Issue: Slow inference

**Solution:**

```python
# 1. Use quantization
model = quantize_model(model, bits=4)

# 2. Reduce max_tokens
output = generate(model, tokenizer, prompt, max_tokens=50)  # Instead of 100+

# 3. Use batch processing for multiple prompts
```

### Issue: Import errors

**Solution:**

```bash
# Reinstall with all dependencies
pip install -e ".[all]"

# Or install specific groups
pip install -e ".[dev,evals,server]"
```

## Next Steps

### Learn More

- **[Model Implementations](docs/ModelImplementations.md)** - Learn how to add new models
- **[Server API](docs/Server.md)** - Deploy production APIs
- **[Agent System](docs/Agents.md)** - Build intelligent agents
- **[Quantization](docs/Quant.md)** - Optimize memory usage

### Explore Advanced Features

1. **Fine-tuning with LoRA**

   ```bash
   python examples/quant/lora_example.py
   ```

2. **Multi-agent collaboration**

   ```bash
   python examples/agents/multi_agent_example.py
   ```

3. **Custom evaluation benchmarks**

   ```bash
   python examples/eval/custom_eval.py
   ```

4. **Production deployment**

   ```bash
   docker build -t smlx-api .
   docker run -p 8000:8000 smlx-api
   ```

### Join the Community

- **GitHub Issues** - Report bugs and request features
- **Discussions** - Ask questions and share projects
- **Contributing** - Help improve SMLX

## Quick Reference

### Model Loading

```python
# Language models
from smlx.models.SmolLM2_135M import load
model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

# Vision-language models
from smlx.models.SmolVLM_256M import load
model, processor = load("HuggingFaceTB/SmolVLM-256M-Instruct")

# Audio models
from smlx.models.Whisper_tiny import load
model, processor = load()

# Document models
from smlx.models.TrOCR_small import load
model, processor = load("printed")  # or "handwritten"

# Embedding models
from smlx.models.MiniLM import load
model, tokenizer = load()
```

### Generation Functions

```python
# Text generation
from smlx.models.SmolLM2_135M import generate, chat, stream_generate

# Basic generation
output = generate(model, tokenizer, prompt, max_tokens=100)

# Streaming
for chunk in stream_generate(model, tokenizer, prompt):
    print(chunk, end="", flush=True)

# Chat
messages = [{"role": "user", "content": "Hello"}]
response = chat(model, tokenizer, messages)
```

### Quantization

```python
from smlx.quant import quantize_model

# 4-bit quantization (75% memory reduction)
model = quantize_model(model, bits=4, group_size=64)

# 8-bit quantization (50% memory reduction)
model = quantize_model(model, bits=8, group_size=128)
```

### Agent Tools

```python
from smlx.agents import ReActAgent, CoTAgent, SelfConsistencyCoTAgent
from smlx.agents.tools import ToolRegistry, calculator, get_time, wikipedia_search

# Create registry
registry = ToolRegistry()
registry.register(calculator)
registry.register(get_time)

# Create agent
agent = ReActAgent(model, tokenizer, registry)
response = agent.run("What is 15 * 23?")
```

---

**You're ready to start building with SMLX!** 🚀

For questions or issues, please visit: <https://github.com/yourusername/smlx/issues>
