# 🔊 Smol Audio

Practical notebooks for shrinking, optimizing, and customizing audio AI models with the Hugging Face ecosystem.

### Latest examples
- Dialogue TTS with Dia-1.6B
- Controllable TTS with Parler-TTS
- Audio understanding with Qwen2-Audio

> [!NOTE]
> GitHub doesn't always render notebooks well. If you have trouble viewing them, try opening in Colab using the links below.

| Category | Notebook | Description |
|----------|----------|-------------|
| ASR Fine-tuning | [Fine-tune Whisper](Fine_tune_Whisper.ipynb) | Fine-tune Whisper on a custom language/domain using transformers + datasets |
| Speed-up | [Faster Whisper with torch.compile](Faster_Whisper_with_torch_compile.ipynb) | Improve Whisper inference latency with `torch.compile` |
| Quantization/ONNX | [Quantize Whisper with Optimum](Quantize_Whisper_with_Optimum.ipynb) | Export Whisper to ONNX and quantize with Optimum |
| TTS Fine-tuning | [Fine-tune SpeechT5](Fine_tune_SpeechT5_TTS.ipynb) | Fine-tune SpeechT5 for custom voice text-to-speech |
| TTS Inference | [Dia TTS](Dia_TTS_Inference.ipynb) | Generate realistic dialogue speech with Dia-1.6B |
| TTS Inference | [Parler-TTS](Parler_TTS_Inference.ipynb) | Controllable TTS with natural language voice descriptions |
| Audio Understanding | [Qwen2-Audio](Qwen2_Audio_Understanding.ipynb) | Audio understanding and QA with Qwen2-Audio |
| Audio Classification | [Audio Classification Fine-tune](Audio_Classification_Fine_tune.ipynb) | Fine-tune Wav2Vec2/HuBERT for audio classification |
| Speech-to-Speech | [Speech-to-Speech Pipeline](Speech_to_Speech_Pipeline.ipynb) | End-to-end pipeline combining ASR + TTS (Whisper + Parler-TTS) |
| Audio Captioning | [Fine-tune Audio Flamingo 3](Fine_tune_Audio_Flamingo_3.ipynb) | Fine-tune Audio Flamingo 3 for audio captioning (full + LoRA) |
| ASR Fine-tuning | [Fine-tune Parakeet](Fine_tune_Parakeet.ipynb) | Fine-tune NVIDIA Parakeet CTC for speech recognition (full + LoRA) |
| ASR Fine-tuning | [Fine-tune Voxtral ASR](Fine_tune_Voxtral_ASR.ipynb) | Fine-tune Voxtral for ASR with prompt masking (full + LoRA) |
| TTS Fine-tuning | [Fine-tune Dia TTS](Fine_tune_Dia_TTS.ipynb) | Fine-tune Dia 1.6B on multi-speaker conversational data |
