# Cohere Transcribe streaming

Run [Cohere Streaming](https://huggingface.co/Steveeeeeeen/cohere-streaming) or distill a streaming student from [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026).

- [Streaming notebook](Stream_Cohere_Transcribe.ipynb): load a checkpoint and transcribe audio in chunks. [Open in Colab](https://colab.research.google.com/github/Deep-unlearning/smol-audio/blob/master/cohere_transcribe_streaming/Stream_Cohere_Transcribe.ipynb).
- [Distillation notebook](Distill_Streaming_Student.ipynb): generate teacher labels, train with `transformers.Trainer`, and evaluate a student. [Open in Colab](https://colab.research.google.com/github/Deep-unlearning/smol-audio/blob/master/cohere_transcribe_streaming/Distill_Streaming_Student.ipynb).

## Setup

Use Python 3.10 or newer. Accept access to the [gated base model](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026), then authenticate with the same Hugging Face account:

```bash
cd cohere_transcribe_streaming
pip install -r requirements.txt
hf auth login
```

A GPU is recommended for inference. Training the 2B model requires substantially more memory; the distillation notebook explains the hardware requirements.

## Code

- `model.py`: streaming encoder, weight conversion, and `CohereStreaming.reset()`, `step()`, and `finish()`.
- `distill_dataset.py`: generate teacher transcripts and save a training dataset.
- `train.py`: distill the student and save checkpoints. Metrics use [Trackio](https://huggingface.co/docs/trackio/transformers_integration).

Adapted from [cohere-transcribe-streaming PR #3](https://github.com/Deep-unlearning/cohere-transcribe-streaming/pull/3), revision [`1e1b41c`](https://github.com/Deep-unlearning/cohere-transcribe-streaming/tree/1e1b41ca259b85d0e2b257b6a794d1b9c1e1ef2d). Code is Apache-2.0; see [LICENSE](../LICENSE). Training datasets retain their own licenses.
