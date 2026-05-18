# Fine-tune NeuTTS Nano to a new language

Tutorial + companion scripts for adapting [Neuphonic's NeuTTS Nano](https://huggingface.co/neuphonic/neutts-nano) to a non-English language using the upstream-style SFTTrainer flow.

The recipe is language-agnostic. Italian is used as the worked example because there is a pre-encoded dataset on the Hub and a validated config in [`../configs/`](../configs).

## Contents

| File | What it is |
|---|---|
| [`Fine_tune_NeuTTS_Nano_New_Language.ipynb`](Fine_tune_NeuTTS_Nano_New_Language.ipynb) | The tutorial. Self-contained, Colab-friendly. Unpacks phonemization, the chat template, label masking, and SFTTrainer setup. |
| [`finetune_neutts_nano.py`](finetune_neutts_nano.py) | Production CLI: reads an OmegaConf YAML, runs the same flow as the notebook, optionally pushes the checkpoint to the Hub. |
| [`encode_dataset.py`](encode_dataset.py) | Minimal CLI: source dataset with text + audio columns → NeuCodec tokens → `text` + `codes` dataset saved locally or pushed to the Hub. |
| [`generate_samples.py`](generate_samples.py) | Slim voice-clone inference CLI using Neuphonic's official `neutts.NeuTTS` engine. |
| [`config_yodas_it.yaml`](config_yodas_it.yaml) | Example training config matching the validated 300k Italian YODAS-Granary run. |

## Typical pipeline

```bash
# 1. (Optional) Encode your own audio into NeuCodec tokens.
python encode_dataset.py \
    --source-dataset nvidia/Granary --source-config it_voxpopuli --source-split asr \
    --output-dataset <your-user>/granary-it-neucodec

# 2. Train. Edit config_yodas_it.yaml to point at your dataset, then:
python finetune_neutts_nano.py config_yodas_it.yaml

# 3. Generate voice-cloned samples from the trained checkpoint.
python generate_samples.py \
    --model-checkpoint ./runs/neutts-official-yodas-300k-5s30s/yodas-300k-5s30s-official-trainer-it-b128-gc \
    --encoded-dataset <your-user>/granary-it-neucodec \
    --reference-index 0 \
    --language it \
    --prompts "Ciao mondo." "Oggi piove molto."
```

The notebook walks through the same flow inline, with explanations of each step. Open it in Colab if you prefer a step-by-step run.

## Adapting to another language

Change two things:

1. `phonemizer_lang` in the YAML (or `PHONEMIZER_LANG` in the notebook) — any [eSpeak language code](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md).
2. `dataset` — a Hub dataset with `text` and `codes` columns. If you don't have one, run `encode_dataset.py` on a public ASR dataset first.

## Hardware

Smoke runs work on a single 16 GB GPU. The full 3000-step Italian config in `config_yodas_it.yaml` is sized for an 80 GB A100 / H100. Lower `per_device_train_batch_size` and re-enable `gradient_checkpointing` if you have less memory.

## Naming note

There is also a `finetune_neutts_nano.py` script at the repo **root** — same basename, different file. The root script is a heavier multi-purpose tool (voice cloning, raw-audio encoding, Whisper WER eval). The one in this folder is the simpler official-trainer flow. They are independent; this folder does not depend on the root script.
