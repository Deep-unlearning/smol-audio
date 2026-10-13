# Cohere Transcribe streaming

Run [Cohere Streaming](https://huggingface.co/Steveeeeeeen/cohere-streaming) or distill a streaming student from [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026).

The Hub checkpoint includes its streaming code and processor. Load it with Transformers:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("Steveeeeeeen/cohere-streaming", trust_remote_code=True)
model.reset(language="en")
for chunk in audio_chunks:  # Mono float32 NumPy audio, 16 kHz, 1,280 samples per chunk.
    print(model.step(chunk))
print(model.finish())
```

The loader chooses CUDA, Apple Metal, or CPU. `device`, `dtype`, and `lookahead` can be passed explicitly. The streaming API is `reset()`, `step()`, and `finish()`.

- [Streaming notebook](Stream_Cohere_Transcribe.ipynb): load a checkpoint and transcribe audio in chunks. [Open in Colab](https://colab.research.google.com/github/Deep-unlearning/smol-audio/blob/master/cohere_transcribe_streaming/Stream_Cohere_Transcribe.ipynb).
- [Distillation notebook](Distill_Streaming_Student.ipynb): generate teacher labels, train with `transformers.Trainer`, and evaluate a student. [Open in Colab](https://colab.research.google.com/github/Deep-unlearning/smol-audio/blob/master/cohere_transcribe_streaming/Distill_Streaming_Student.ipynb).

## Setup

Use Python 3.10 or newer. Training and teacher labeling require access to the [gated base model](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026). Authenticate with an account that can access the checkpoints you use:

```bash
cd cohere_transcribe_streaming
pip install -r requirements.txt
hf auth login
```

A GPU is recommended for inference. Training the 2B model requires substantially more memory; the distillation notebook explains the hardware requirements.

## Code

- `model.py`: streaming encoder, Cohere weight conversion, and Transformers remote-code loading. The local `CohereStreaming` adapter remains available.
- `distill_dataset.py`: generate teacher transcripts and save a training dataset.
- `train.py`: distill the student and save checkpoints. Metrics use [Trackio](https://huggingface.co/docs/trackio/transformers_integration).

Adapted from [cohere-transcribe-streaming PR #3](https://github.com/Deep-unlearning/cohere-transcribe-streaming/pull/3), revision [`1e1b41c`](https://github.com/Deep-unlearning/cohere-transcribe-streaming/tree/1e1b41ca259b85d0e2b257b6a794d1b9c1e1ef2d). Code is Apache-2.0; see [LICENSE](../LICENSE). Training datasets retain their own licenses.
