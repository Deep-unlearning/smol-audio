<p align="center">
  <img src="image.png" alt="Smol Audio" />
</p>

# Smol Audio 🔊

Practical notebooks for shrinking, optimizing, and customizing audio AI models with the Hugging Face ecosystem.

### Latest examples
- Stream Cohere Transcribe and distill your own streaming student
- Inference with Perception Encoder for Audio-Video (PE-AV)
- Fine-tune Audio Flamingo 3
- Granite Speech 4.0 1b ASR

> [!NOTE]
> GitHub doesn't always render notebooks well. If you have trouble viewing them, try opening in Colab using the links below.

| Category | Notebook | Description |
|----------|----------|-------------|
| Streaming ASR | [Stream Cohere Transcribe](cohere_transcribe_streaming/Stream_Cohere_Transcribe.ipynb) · [Colab](https://colab.research.google.com/github/Deep-unlearning/smol-audio/blob/master/cohere_transcribe_streaming/Stream_Cohere_Transcribe.ipynb) | Run the released streaming model on 80 ms audio chunks and compare lookahead settings |
| ASR Distillation | [Distill a streaming student](cohere_transcribe_streaming/Distill_Streaming_Student.ipynb) · [Colab](https://colab.research.google.com/github/Deep-unlearning/smol-audio/blob/master/cohere_transcribe_streaming/Distill_Streaming_Student.ipynb) | Teacher-label a small dataset, train a streaming Cohere student, evaluate, and stream with the checkpoint; [model and training code](cohere_transcribe_streaming/) |
| ASR Fine-tuning | [Fine-tune Whisper](Fine_tune_Whisper.ipynb) | Fine-tune Whisper on a custom language/domain using transformers + datasets |
| ASR Fine-tuning | [Fine-tune Granite Speech Italian](Fine_tune_Granite_Speech_Italian.ipynb) | Fine-tune IBM Granite Speech for Italian ASR with the YODAS-Granary dataset |
| Audio Captioning | [Fine-tune Audio Flamingo 3](Fine_tune_Audio_Flamingo_3.ipynb) | Fine-tune Audio Flamingo 3 for audio captioning (full + LoRA) |
| ASR Fine-tuning | [Fine-tune Parakeet](Fine_tune_Parakeet.ipynb) | Fine-tune NVIDIA Parakeet CTC for speech recognition (full + LoRA) |
| ASR Fine-tuning | [Fine-tune Voxtral ASR](Fine_tune_Voxtral_ASR.ipynb) | Fine-tune Voxtral for ASR with prompt masking (full + LoRA) |
| Multimodal | [Inference with PE-AV-Base](Inference_PE_AV_Base.ipynb) | Zero-shot video classification and audio↔text retrieval (AudioCaps) with Meta's Perception Encoder for Audio-Video |
