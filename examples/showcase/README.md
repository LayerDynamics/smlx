# SMLX Showcase Examples

End-to-end demos that pair **the datasets bundled in `data/`** with **real smol
models**, each producing a real result or measured metric — entirely on-device.
Every demo loads its data through `smlx.data.local`, so no dataset download is
required (the data already ships in the repo).

| Demo | Modality | Model | Local data | What it shows |
|------|----------|-------|-----------|---------------|
| [`minilm_semantic_search.py`](minilm_semantic_search.py) | Embeddings | all-MiniLM-L6-v2 (~23M) | `data/benchmark/glue` | Semantic search by cosine similarity |
| [`smolvlm_coco8_vqa.py`](smolvlm_coco8_vqa.py) | Vision-language | SmolVLM-256M-Instruct | `data/datasets/coco8` | Captioning + visual Q&A |
| [`whisper_librispeech_wer.py`](whisper_librispeech_wer.py) | Speech | Whisper-tiny (~39M) | `data/audio/speech/librispeech_sample` | ASR with measured WER/CER |
| [`smollm2_wikitext_perplexity.py`](smollm2_wikitext_perplexity.py) | Language | SmolLM2-135M-Instruct | `data/benchmark/wikitext` | Language-model perplexity |

## Running

Use the project's Python environment:

```bash
conda activate smlx

python examples/showcase/minilm_semantic_search.py --query "a film worth watching"
python examples/showcase/smolvlm_coco8_vqa.py --num-images 3
python examples/showcase/whisper_librispeech_wer.py --num-clips 5
python examples/showcase/smollm2_wikitext_perplexity.py --max-sequences 30
```

The embedding and VLM demos use models that are typically already cached; the
Whisper and SmolLM2 demos download their (small) models from the Hub on first
run.

## Verified output (reference)

These are representative results from a local run, so you know what "working"
looks like:

- **Semantic search** — query `"a film worth watching"` ranks
  `"a sometimes tedious film ."` and `"a gorgeous , witty , seductive movie ."`
  at the top.
- **SmolVLM / COCO8** — image `000000000025.jpg` →
  *"There are two giraffes on the left side of the image eating from a tree…"*
- **Whisper / LibriSpeech** — ~**10% WER**, ~**4% CER** over 4 clips.
- **SmolLM2 / WikiText-2** — perplexity ≈ **26** over 10k tokens.

## Inspecting the data first

Before (or instead of) loading a model, browse the bundled datasets:

```bash
python scripts/validate_data.py            # health-check every dataset
python scripts/preview_data.py coco8        # peek at samples
# or via the CLI
python -m smlx.main data list
python -m smlx.main data preview glue --split sst2_validation -n 3
```
