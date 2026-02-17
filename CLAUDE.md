# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

A cookbook of self-contained Jupyter notebooks for shrinking, optimizing, and customizing audio AI models using the Hugging Face ecosystem. Modeled after [smol-vision](https://github.com/merveenoyan/smol-vision).

## Structure

Flat repo — all notebooks live at the root. No shared Python modules or build system. Each notebook is independent and runnable on Google Colab without local setup.

Categories: ASR fine-tuning, inference speed-up, ONNX quantization, TTS fine-tuning, TTS inference, audio understanding, audio classification, speech-to-speech pipelines.

## Conventions

- Notebook filenames use `Title_Case_With_Underscores.ipynb`
- Every notebook starts with a pip install cell, then load data/model, then task-specific steps, then inference/evaluation
- README.md contains the master table linking all notebooks — update it when adding/removing notebooks
- License is Apache-2.0

## Validating Notebooks

```bash
python3 -c "import json; [json.load(open(f)) for f in __import__('glob').glob('*.ipynb')]"
```
