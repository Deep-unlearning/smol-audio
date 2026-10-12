---
title: Streaming Cohere Transcribe
emoji: 🎙️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: 3.12.12
app_file: app.py
license: apache-2.0
models:
  - Steveeeeeeen/cohere-streaming
  - CohereLabs/cohere-transcribe-03-2026
tags:
  - automatic-speech-recognition
  - streaming
---

# Streaming Cohere Transcribe

Choose a language, click **Record**, and speak. The transcript updates while you record; **Stop** flushes the remaining audio and finalizes the transcript. Clips are limited to 25 seconds. **Clear** resets the demo.

The browser captures mono audio and sends a chunk every 400 ms. The server resamples to 16 kHz, feeds 80 ms steps into [Cohere Streaming](https://huggingface.co/Steveeeeeeen/cohere-streaming), and returns partial transcripts. Final text can revise earlier words. Audio is temporarily stored on this Space and removed after processing.

This Space uses free shared ZeroGPU hardware. GPU queues and daily quotas can delay or limit recording. Start another recording if a request fails.

## Run locally

From `cohere_transcribe_streaming/space` in [smol-audio](https://github.com/Deep-unlearning/smol-audio):

```bash
pip install gradio==6.30.0 -r requirements.txt
python app.py
```

A CUDA GPU is recommended. Accept access to [the base Cohere model](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) and run `hf auth login` before loading it. The two-billion-parameter model also needs enough host memory to load the base model and streaming checkpoint. Local CPU inference is slower.

## Deploy your own Space

Create a Gradio Space with ZeroGPU hardware. Copy `app.py`, `recorder.html`, `recorder.js`, `requirements.txt`, and this `README.md` into its root. Add `model.py` from the parent example directory and `LICENSE` from the repository root. Add an `HF_TOKEN` Space secret with read access to the gated base Cohere model. Visitors use the demo without providing a token.

The Python app keeps one GPU worker active for each recording. CPU callbacks write ordered microphone chunks to temporary files, which that worker consumes without releasing its streaming caches between chunks. The browser recorder explicitly sends its last buffered samples before signaling Stop.
