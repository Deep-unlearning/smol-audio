import os
import re
import warnings
import argparse
import inspect
import logging
from functools import partial

import phonemizer
import torch
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)
from trl import SFTTrainer, SFTConfig

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)
PHONEMIZER_LOGGER = logging.getLogger("phonemizer")
PHONEMIZER_LOGGER.setLevel(logging.ERROR)


ACRONYM = re.compile(r"(?:[a-zA-Z]\.){2,}")
ACRONYM_NO_PERIOD = re.compile(r"(?:[A-Z]){2,}")
IGNORE_INDEX = -100


def cfg_get(config, name, default=None):
    return getattr(config, name) if name in config else default


def data_filter(sample):
    text = sample["text"]

    if len(text) == 0:
        return False

    if re.search(r"\d", text):
        return False

    if re.search(ACRONYM, text) or re.search(ACRONYM_NO_PERIOD, text):
        return False

    if text[-1] not in ".,?!":
        return False

    if "£" in text or "$" in text:
        return False

    return True


def empty_features():
    return {
        "input_ids": None,
        "labels": None,
        "attention_mask": None,
    }


def preprocess_sample(sample, tokenizer, max_len, g2p, language_token=""):
    speech_gen_start = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
    speech_gen_end = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

    vq_codes = sample["codes"]
    text = sample["text"]

    phones = g2p.phonemize([text])
    if not phones or not phones[0]:
        LOGGER.warning("Empty phonemization output for text=%r", text)
        return empty_features()

    phones = " ".join(phones[0].split())
    if language_token:
        phones = f"{language_token} {phones}"

    codes_str = "".join([f"<|speech_{i}|>" for i in vq_codes])
    chat = (
        f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{phones}<|TEXT_PROMPT_END|>\n"
        f"assistant:<|SPEECH_GENERATION_START|>{codes_str}<|SPEECH_GENERATION_END|>"
    )
    ids = tokenizer.encode(chat)

    if len(ids) > max_len:
        LOGGER.warning("Dropping overlength sample: %d tokens > max_seq_len=%d", len(ids), max_len)
        return empty_features()

    if len(ids) < max_len:
        ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))

    input_ids = torch.tensor(ids, dtype=torch.long)
    labels = torch.full_like(input_ids, IGNORE_INDEX)

    speech_gen_start_idx = (input_ids == speech_gen_start).nonzero(as_tuple=True)[0]
    if len(speech_gen_start_idx) > 0:
        speech_gen_start_idx = speech_gen_start_idx[0]
        speech_gen_end_idx = (input_ids == speech_gen_end).nonzero(as_tuple=True)[0]
        if len(speech_gen_end_idx) == 0:
            LOGGER.warning("Dropping sample with no speech end token after tokenization.")
            return empty_features()
        speech_gen_end_idx = speech_gen_end_idx[0]
        labels[speech_gen_start_idx : speech_gen_end_idx + 1] = input_ids[speech_gen_start_idx : speech_gen_end_idx + 1]
    else:
        LOGGER.warning("Dropping sample with no speech start token after tokenization.")
        return empty_features()

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def main(config_fpath: str):
    print(f"Loading config from {config_fpath}")
    config = OmegaConf.load(config_fpath)
    checkpoints_dir = os.path.join(config.save_root, config.run_name)
    LOGGER.info("Logging to: %s", checkpoints_dir)

    restore_from = config.restore_from
    print(f"Loading checkpoint from {restore_from}")
    tokenizer = AutoTokenizer.from_pretrained(restore_from)
    model = AutoModelForCausalLM.from_pretrained(restore_from, torch_dtype="auto")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    tokenizer.padding_side = "right"

    language_token = cfg_get(config, "language_token", "")
    if language_token and tokenizer.convert_tokens_to_ids(language_token) == tokenizer.unk_token_id:
        tokenizer.add_tokens([language_token])
        if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    phonemizer_lang = cfg_get(config, "phonemizer_lang", "en-us")
    g2p = phonemizer.backend.EspeakBackend(
        language=phonemizer_lang,
        preserve_punctuation=True,
        with_stress=True,
        words_mismatch="ignore",
        language_switch="remove-flags",
        logger=PHONEMIZER_LOGGER,
    )

    partial_preprocess = partial(
        preprocess_sample,
        tokenizer=tokenizer,
        max_len=config.max_seq_len,
        g2p=g2p,
        language_token=language_token,
    )

    dataset_name = cfg_get(config, "dataset", "neuphonic/emilia-yodas-english-neucodec")
    dataset_split = cfg_get(config, "split", "train[:2000]")
    num_proc = cfg_get(config, "preprocess_num_proc", None)
    remove_columns = cfg_get(config, "remove_columns", ["text", "codes"])

    print(f"Loading dataset {dataset_name} split={dataset_split}")
    train_dataset = load_dataset(dataset_name, split=dataset_split)
    print(f"Rows before filter: {len(train_dataset)}")
    train_dataset = train_dataset.filter(data_filter)
    print(f"Rows after filter: {len(train_dataset)}")
    train_dataset = train_dataset.map(
        partial_preprocess,
        remove_columns=remove_columns,
        num_proc=num_proc,
    )
    train_dataset = train_dataset.filter(lambda sample: sample["input_ids"] is not None)
    print(f"Rows after preprocessing: {len(train_dataset)}")

    training_args = SFTConfig(
        output_dir=checkpoints_dir,
        do_train=True,
        learning_rate=config.lr,
        max_steps=config.max_steps,
        bf16=cfg_get(config, "bf16", True),
        per_device_train_batch_size=config.per_device_train_batch_size,
        warmup_ratio=config.warmup_ratio,
        gradient_accumulation_steps=cfg_get(config, "gradient_accumulation_steps", 1),
        gradient_checkpointing=cfg_get(config, "gradient_checkpointing", True),
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        ignore_data_skip=cfg_get(config, "ignore_data_skip", True),
        dataloader_drop_last=cfg_get(config, "dataloader_drop_last", True),
        remove_unused_columns=False,
        torch_compile=cfg_get(config, "torch_compile", False),
        dataloader_num_workers=cfg_get(config, "dataloader_num_workers", 64),
        save_total_limit=cfg_get(config, "save_total_limit", None),
        report_to=cfg_get(config, "report_to", "none"),
        loss_type="chunked_nll",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer_kwargs = {}
    trainer_signature = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        **trainer_kwargs,
    )

    resume_from_checkpoint = cfg_get(config, "resume_from_checkpoint", None)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(checkpoints_dir)
    tokenizer.save_pretrained(checkpoints_dir)

    hub_model_id = cfg_get(config, "hub_model_id", None)
    if hub_model_id:
        hub_private = bool(cfg_get(config, "hub_private", False))
        LOGGER.info("Uploading final model to Hub model repo: %s", hub_model_id)
        create_repo(hub_model_id, repo_type="model", private=hub_private, exist_ok=True)
        upload_folder(
            repo_id=hub_model_id,
            repo_type="model",
            folder_path=checkpoints_dir,
            path_in_repo=".",
            commit_message=f"Upload {cfg_get(config, 'run_name', 'NeuTTS official trainer run')}",
        )
        LOGGER.info("Uploaded final model to: https://huggingface.co/%s", hub_model_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuTTS with the upstream-style plain Trainer flow.")
    parser.add_argument("config_fpath", help="Path to an OmegaConf/YAML config file.")
    args = parser.parse_args()
    main(args.config_fpath)
