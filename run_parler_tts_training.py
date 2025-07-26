#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Train Parler-TTS using 🤗 Accelerate"""

import logging
import os
import re
import sys
import time
import math
import contextlib
from multiprocess import set_start_method
from datetime import timedelta
import inspect
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import datasets
from datasets import DatasetDict, Dataset, IterableDataset, concatenate_datasets

from huggingface_hub import HfApi

import transformers
from transformers import AutoFeatureExtractor, AutoTokenizer, HfArgumentParser
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.optimization import get_scheduler
from transformers.utils import send_example_telemetry


from accelerate import Accelerator, skip_first_batches
from accelerate.utils import set_seed, AutocastKwargs, InitProcessGroupKwargs, TorchDynamoPlugin, DistributedDataParallelKwargs
from accelerate.utils.memory import release_memory

from parler_tts import (
    ParlerTTSConfig,
    ParlerTTSForConditionalGeneration,
    build_delay_pattern_mask,
)

from utils import (
    get_last_checkpoint,
    rotate_checkpoints,
    log_pred,
    log_metric,
    load_all_codec_checkpoints,
    save_codec_checkpoint,
    get_last_codec_checkpoint_step,
)
from arguments import ModelArguments, DataTrainingArguments, ParlerTTSTrainingArguments
from data import load_multiple_datasets, DataCollatorParlerTTSWithPadding, DataCollatorEncodecWithPadding
from eval import clap_similarity, wer, si_sdr

# Hardy: I added some new imports here
import json
import numpy as np
from eval import clap_similarity, wer, si_sdr, speech_emotion_recognition

# Hardy: Add this flag to track if we've done the feasibility test
# Hardy: Change its location inside main() due to the UnboundLocalError.
# feasibility_test_done = False

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # Hardy: Change its location inside main() due to the UnboundLocalError.
    feasibility_test_done = False

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ParlerTTSTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_parler_tts", model_args, data_args)

    if training_args.dtype == "float16":
        mixed_precision = "fp16"
        torch_dtype = torch.float16
    elif training_args.dtype == "bfloat16":
        mixed_precision = "bf16"
        torch_dtype = torch.bfloat16
    else:
        mixed_precision = "no"
        torch_dtype = torch.float32

    if data_args.pad_to_max_length and (
        data_args.max_duration_in_seconds is None
        or data_args.max_prompt_token_length is None
        or data_args.max_description_token_length is None
    ):
        raise ValueError(
            "`pad_to_max_length` is `True` but one of the following parameters has not been set: `max_duration_in_seconds`, `max_prompt_token_length`, `max_description_token_length`"
        )

    padding = "max_length" if data_args.pad_to_max_length else "longest"

    ####### A. Preparation
    kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(minutes=120)), DistributedDataParallelKwargs(find_unused_parameters=False)]

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
        kwargs_handlers=kwargs_handlers,
    )

    accelerator.init_trackers(
        project_name=data_args.wandb_project,
        config={
            "learning_rate": training_args.learning_rate,
            "model_name_or_path": model_args.model_name_or_path,
            "num_train_epochs": training_args.num_train_epochs,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "global_batch_size": training_args.per_device_train_batch_size * accelerator.num_processes,
            "mixed_precision": mixed_precision,
            "lr_scheduler_type": training_args.lr_scheduler_type,
            "warmup_steps": training_args.warmup_steps,
            "freeze_text_encoder": model_args.freeze_text_encoder,
            "max_duration_in_seconds": data_args.max_duration_in_seconds,
            "weight_decay": training_args.weight_decay,
            "adam_beta1": training_args.adam_beta1,
            "adam_beta2": training_args.adam_beta2,
            "temperature": model_args.temperature,
        },
        init_kwargs={"wandb": {"name": data_args.wandb_run_name}} if data_args.wandb_run_name else {},
    )

    # Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Hardy: Important! Temporarily force all processes to log at INFO level
    logger.setLevel(logging.INFO)
    # Hardy: The original logger setup:
    # logger.setLevel(logging.INFO if accelerator.is_main_process else logging.WARN)

    # Log a small summary on each proces
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)
    num_workers = data_args.preprocessing_num_workers

    # 1. First, lett's instantiate the feature extractor, tokenizers and model
    # Note for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.

    # load feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )
    sampling_rate = feature_extractor.sampling_rate

    # load prompt tokenizer
    prompt_tokenizer = AutoTokenizer.from_pretrained(
        model_args.prompt_tokenizer_name or model_args.description_tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        use_fast=model_args.use_fast_tokenizer,
        padding_side=model_args.prompt_padding_side,
    )

    # load description tokenizer
    description_tokenizer = AutoTokenizer.from_pretrained(
        model_args.description_tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        use_fast=model_args.use_fast_tokenizer,
    )

    if model_args.use_fast_tokenizer:
        logger.warning(
            "Disabling fast tokenizer warning: https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L3231-L3235"
        )
        prompt_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        description_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    # 2. Now, let's load the dataset

    if data_args.save_to_disk is not None:
        os.makedirs(data_args.save_to_disk, exist_ok=True)

    # assume that the dataset has been saved to `save_to_disk` if the latter is not empty
    dataset_was_precomputed = len(os.listdir(data_args.save_to_disk)) > 0
    if dataset_was_precomputed:
        with accelerator.local_main_process_first():
            vectorized_datasets = datasets.load_from_disk(data_args.save_to_disk)
    else:
        raw_datasets = DatasetDict()

        columns_to_keep = {
            "target_audio_column_name": data_args.target_audio_column_name,
            "prompt_column_name": data_args.prompt_column_name,
        }
        if data_args.description_column_name is not None:
            columns_to_keep["description_column_name"] = data_args.description_column_name
        # Hardy: Add a new if condition for the emotion column
        if data_args.emotion_column_name is not None:
            columns_to_keep["emotion_column_name"] = data_args.emotion_column_name

        if training_args.do_train:
            raw_datasets["train"] = load_multiple_datasets(
                accelerator,
                data_args.train_dataset_name,
                data_args.train_dataset_config_name,
                metadata_dataset_names=data_args.train_metadata_dataset_name,
                splits=data_args.train_split_name,
                dataset_samples=data_args.train_dataset_samples,
                seed=training_args.seed,
                cache_dir=model_args.cache_dir,
                num_proc=data_args.preprocessing_num_workers,
                id_column_name=data_args.id_column_name,
                columns_to_keep=columns_to_keep.values(),
                prompt_column_name=data_args.prompt_column_name,
                audio_column_name=data_args.target_audio_column_name,
                sampling_rate=sampling_rate,
                logger=logger,
                # streaming=data_args.streaming, TODO(SG): optionally enable streaming mode
            )
            # Hardy: I added a debug logging here:
            logger.info(f"DEBUG: Train dataset size after loading: {len(raw_datasets['train'])}")

            for key in columns_to_keep:
                if columns_to_keep[key] not in raw_datasets["train"].column_names:
                    raise ValueError(
                        f"--{key} '{columns_to_keep[key]}' not found in dataset '{data_args.train_dataset_name}'."
                        f" Make sure to set `--{key}` to the correct audio column - one of"
                        f" {', '.join(raw_datasets['train'].column_names)}."
                    )

            if data_args.max_train_samples is not None:
                raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))
            # Hardy: I added a debug logging here:
            logger.info(f"DEBUG: Train dataset size after sampling: {len(raw_datasets['train'])}")

        # Hardy: I took some modification here:
        # The original one:
        # if training_args.do_eval:
        #     raw_datasets["eval"] = load_multiple_datasets(
        #         accelerator,
        #         data_args.eval_dataset_name if data_args.eval_dataset_name else data_args.train_dataset_name,
        #         data_args.eval_dataset_config_name
        #         if data_args.eval_dataset_config_name
        #         else data_args.train_dataset_config_name,
        #         metadata_dataset_names=data_args.eval_metadata_dataset_name,
        #         splits=data_args.eval_split_name,
        #         cache_dir=model_args.cache_dir,
        #         num_proc=data_args.preprocessing_num_workers,
        #         id_column_name=data_args.id_column_name,
        #         columns_to_keep=columns_to_keep.values(),
        #         prompt_column_name=data_args.prompt_column_name,
        #         audio_column_name=data_args.target_audio_column_name,
        #         sampling_rate=sampling_rate,
        #         logger=logger,
        #         # streaming=data_args.streaming, TODO(SG): optionally enable streaming mode
        #     )
        # The current one:
        # Replace the evaluation dataset loading section with:
        if training_args.do_eval:
            # Check if eval dataset is a local pre-saved dataset
            if os.path.exists(data_args.eval_dataset_name) and os.path.isdir(data_args.eval_dataset_name):
                # Load pre-saved dataset directly
                with accelerator.local_main_process_first():
                    eval_dataset_dict = datasets.load_from_disk(data_args.eval_dataset_name)
                    # Get the first split if it's a DatasetDict
                    if isinstance(eval_dataset_dict, datasets.DatasetDict):
                        # Use the split specified or default to first available split
                        if data_args.eval_split_name in eval_dataset_dict:
                            raw_datasets["eval"] = eval_dataset_dict[data_args.eval_split_name]
                        else:
                            # Use the first available split
                            first_split = list(eval_dataset_dict.keys())[0]
                            raw_datasets["eval"] = eval_dataset_dict[first_split]
                            logger.info(
                                f"Eval split '{data_args.eval_split_name}' not found, using '{first_split}' instead")
                    else:
                        raw_datasets["eval"] = eval_dataset_dict
            else:
                # Original loading logic for HF hub datasets
                raw_datasets["eval"] = load_multiple_datasets(
                    accelerator,
                    data_args.eval_dataset_name if data_args.eval_dataset_name else data_args.train_dataset_name,
                    data_args.eval_dataset_config_name
                    if data_args.eval_dataset_config_name
                    else data_args.train_dataset_config_name,
                    metadata_dataset_names=data_args.eval_metadata_dataset_name,
                    splits=data_args.eval_split_name,
                    cache_dir=model_args.cache_dir,
                    num_proc=data_args.preprocessing_num_workers,
                    id_column_name=data_args.id_column_name,
                    columns_to_keep=columns_to_keep.values(),
                    prompt_column_name=data_args.prompt_column_name,
                    audio_column_name=data_args.target_audio_column_name,
                    sampling_rate=sampling_rate,
                    logger=logger,
                    # streaming=data_args.streaming, TODO(SG): optionally enable streaming mode
                )

            if data_args.max_eval_samples is not None:
                with accelerator.local_main_process_first():
                    raw_datasets["eval"] = (
                        raw_datasets["eval"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
                    )
            # Hardy: I added a debug logging from here:
            logger.info(f"DEBUG: Eval dataset info:")
            logger.info(f"  - Type: {type(raw_datasets['eval'])}")
            logger.info(f"  - Length: {len(raw_datasets['eval'])}")
            logger.info(
                f"  - Columns: {raw_datasets['eval'].column_names if hasattr(raw_datasets['eval'], 'column_names') else 'No column_names attribute'}")
            if len(raw_datasets['eval']) > 0:
                logger.info(f"  - First example keys: {list(raw_datasets['eval'][0].keys())}")
            # Hardy: to here

    # 3. Next, let's load the config.
    config = ParlerTTSConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )
 
    if training_args.codebook_weights is not None and len(training_args.codebook_weights) != config.decoder.num_codebooks:
        raise ValueError(f"`codebook_weights` has length {len(training_args.codebook_weights)} when it should be of length {config.decoder.num_codebooks}.")

    # update pad token id and decoder_start_token_id
    config.decoder.update(
        {
            "cross_attention_implementation_strategy": model_args.cross_attention_implementation_strategy
            if model_args.cross_attention_implementation_strategy is not None
            else None,
            "codebook_weights": training_args.codebook_weights if training_args.codebook_weights is not None else config.decoder.codebook_weights
        }
    )
    config.update(
        {
            "pad_token_id": model_args.pad_token_id if model_args.pad_token_id is not None else config.pad_token_id,
            "decoder_start_token_id": model_args.decoder_start_token_id
            if model_args.decoder_start_token_id is not None
            else config.decoder_start_token_id,
        }
    )

    # create model
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        attn_implementation={"decoder": model_args.attn_implementation, "text_encoder": "eager"},
    )

    # enable gradient checkpointing if necessary
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 4. Now we preprocess the datasets including loading the audio, resampling and normalization
    # Thankfully, `datasets` takes care of automatically loading and resampling the audio,
    # so that we just need to set the correct target sampling rate and normalize the input
    # via the `feature_extractor`

    # derive max & min input length for sample rate & max duration
    sampling_rate = feature_extractor.sampling_rate
    max_target_length = int(data_args.max_duration_in_seconds * sampling_rate)
    min_target_length = int(data_args.min_duration_in_seconds * sampling_rate)
    target_audio_column_name = data_args.target_audio_column_name
    description_column_name = data_args.description_column_name
    prompt_column_name = data_args.prompt_column_name
    feature_extractor_input_name = feature_extractor.model_input_names[0]
    audio_encoder_pad_token_id = config.decoder.pad_token_id
    audio_encoder_eos_token_id = config.decoder.eos_token_id
    audio_encoder_bos_token_id = model.generation_config.decoder_start_token_id
    max_length = model.generation_config.max_length
    num_codebooks = model.decoder.config.num_codebooks
    bandwidth = model_args.bandwidth
    attn_implementation = model_args.attn_implementation

    # Freeze Encoders
    model.freeze_encoders(model_args.freeze_text_encoder)

    # Test all gather - used for warmout and avoiding timeout
    logger.debug(str(accelerator.process_index), main_process_only=False, in_order=True)
    test_tensor = torch.tensor([accelerator.process_index], device=accelerator.device)
    gathered_tensor = accelerator.gather(test_tensor)
    print("gathered_tensor", gathered_tensor)
    accelerator.wait_for_everyone()

    if not dataset_was_precomputed:
        # Filter on text length
        if description_column_name is not None and data_args.max_text_length is not None:
            with accelerator.local_main_process_first():
                # filter description that is shorter than max_text_length
                raw_datasets = raw_datasets.filter(
                    lambda x: len(x) < data_args.max_text_length,
                    num_proc=num_workers,
                    input_columns=[description_column_name],
                )

        # Preprocessing the dataset.
        # We need to tokenize the texts.
        # Hardy: I modified this:
        # def pass_through_processors(description, prompt):
        #     batch = {}
        #
        #     batch["input_ids"] = description_tokenizer(description.strip())["input_ids"]
        #     batch["prompt_input_ids"] = prompt_tokenizer(prompt.strip())["input_ids"]
        #
        #     return batch
        # Hardy: To this:
        def pass_through_processors(examples):
            """Process text fields and preserve metadata"""
            batch = {}

            # Process text fields
            batch["input_ids"] = description_tokenizer(examples[description_column_name].strip())["input_ids"]
            batch["prompt_input_ids"] = prompt_tokenizer(examples[prompt_column_name].strip())["input_ids"]

            # Preserve metadata fields that exist in the dataset
            metadata_fields_to_preserve = ["style", "gender", "noise", "pitch", "speaking_rate", "test_category",
                                           "number"]
            for field in metadata_fields_to_preserve:
                if field in examples:
                    batch[field] = examples[field]

            return batch

        # Hardy: I changed this --
        # with accelerator.local_main_process_first():
        #     # this is a trick to avoid to rewrite the entire audio column which takes ages
        #     vectorized_datasets = raw_datasets.map(
        #         pass_through_processors,
        #         remove_columns=next(iter(raw_datasets.values())).column_names,
        #         input_columns=[description_column_name, prompt_column_name],
        #         num_proc=num_workers,
        #         desc="preprocess datasets",
        #     )
        # Hardy: To this (updated again):
        with accelerator.local_main_process_first():
            # Process each dataset split separately
            vectorized_datasets = {}
            for split_name, dataset in raw_datasets.items():
                # Debug logging
                logger.info(f"DEBUG: Processing {split_name} dataset")
                logger.info(f"DEBUG:   - Size: {len(dataset)}")
                logger.info(f"DEBUG:   - Columns: {dataset.column_names}")
                if len(dataset) > 0:
                    logger.info(f"DEBUG:   - First example keys: {list(dataset[0].keys())}")
                    # Check if style column exists
                    if "style" in dataset[0]:
                        logger.info(f"DEBUG:   - First example style: {dataset[0]['style']}")

                # Determine which columns to remove (everything except what we need)
                columns_to_keep = {"input_ids", "prompt_input_ids", "style", "gender", "noise",
                                   "pitch", "speaking_rate", "test_category", "number"}
                columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]

                # Map the preprocessing function
                vectorized_datasets[split_name] = dataset.map(
                    pass_through_processors,
                    remove_columns=columns_to_remove,
                    num_proc=num_workers,
                    desc=f"preprocess {split_name} dataset",
                    batched=False,  # Process one example at a time
                )

                # Verify the style column was preserved
                if "style" in vectorized_datasets[split_name].column_names:
                    logger.info(f"DEBUG: Successfully preserved 'style' column in {split_name} dataset")
                else:
                    logger.warning(f"WARNING: 'style' column not found in processed {split_name} dataset!")

            # Convert back to DatasetDict
            vectorized_datasets = DatasetDict(vectorized_datasets)
        # Hardy: up to here

        # We use Accelerate to perform distributed inference
        # T5 doesn't support fp16
        autocast_kwargs = AutocastKwargs(enabled=(mixed_precision != "fp16"))

        # Hardy: The following is the modified Section B of Encode audio
        # Now we encode the audio labels with encodec.
        ####### B. Encode audio

        logger.info("*** Encode target audio with encodec ***")

        # no need to prepare audio_decoder because used for inference without mixed precision
        # see: https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.prepare
        if training_args.torch_compile:
            audio_decoder = accelerator.prepare_model(model.audio_encoder, evaluation_mode=True)
        else:
            audio_decoder = model.audio_encoder

        encoder_data_collator = DataCollatorEncodecWithPadding(
            feature_extractor,
            audio_column_name=target_audio_column_name,
            feature_extractor_input_name=feature_extractor_input_name,
            max_length=max_target_length,
            padding=padding,
        )
        encoder_signature = set(inspect.signature(audio_decoder.forward).parameters)

        def apply_audio_decoder(batch):
            len_audio = batch.pop("len_audio")
            audio_decoder.to(batch["input_values"].device).eval()
            if bandwidth is not None:
                batch["bandwidth"] = bandwidth
            elif "num_quantizers" in encoder_signature:
                batch["num_quantizers"] = num_codebooks
            elif "num_codebooks" in encoder_signature:
                batch["num_codebooks"] = num_codebooks
            elif "n_quantizers" in encoder_signature:
                batch["n_quantizers"] = num_codebooks

            with torch.no_grad():
                labels = audio_decoder.encode(**batch)["audio_codes"]
            output = {}
            output["len_audio"] = len_audio
            # (1, bsz, codebooks, seq_len) -> (bsz, seq_len, codebooks)
            output["labels"] = labels.squeeze(0).transpose(1, 2)

            # if `pad_to_max_length`, the maximum corresponding audio length of the current batch is max_duration*sampling_rate
            max_length = len_audio.max() if padding != "max_length" else max_target_length
            output["ratio"] = torch.ones_like(len_audio) * labels.shape[-1] / max_length
            return output

        # (1, codebooks, seq_len) where seq_len=1
        bos_labels = torch.ones((1, num_codebooks, 1)) * audio_encoder_bos_token_id

        def postprocess_dataset(labels):
            # (1, codebooks, seq_len)
            labels = torch.tensor(labels).unsqueeze(0)
            # add bos
            labels = torch.cat([bos_labels, labels], dim=-1)

            labels, delay_pattern_mask = build_delay_pattern_mask(
                labels,
                bos_token_id=audio_encoder_bos_token_id,
                pad_token_id=audio_encoder_eos_token_id,
                max_length=labels.shape[-1] + num_codebooks,
                num_codebooks=num_codebooks,
            )

            # the first ids of the delay pattern mask are precisely labels, we use the rest of the labels mask
            # to take care of EOS
            # we want labels to look like this:
            #  - [B, a, b, E, E, E, E]
            #  - [B, B, c, d, E, E, E]
            #  - [B, B, B, e, f, E, E]
            #  - [B, B, B, B, g, h, E]
            labels = torch.where(delay_pattern_mask == -1, audio_encoder_eos_token_id, delay_pattern_mask)

            # the first timestamp is associated to a row full of BOS, let's get rid of it
            # we also remove the last timestampts (full of PAD)
            output = {"labels": labels[:, 1:]}
            return output

        for split in vectorized_datasets:
            # Check if this split has audio data
            if target_audio_column_name in raw_datasets[split].column_names:
                # Hardy: I added a debug logging here:
                logger.info(f"DEBUG: Columns in {split} split: {raw_datasets[split].column_names}")

                logger.info(f"Encoding audio for {split} split...")

                data_loader = DataLoader(
                    raw_datasets[split],
                    batch_size=training_args.audio_encoder_per_device_batch_size,
                    collate_fn=encoder_data_collator,
                    num_workers=training_args.dataloader_num_workers,
                    pin_memory=True,
                )
                data_loader = accelerator.prepare(data_loader)
                total_inference_steps = len(data_loader)

                start_step = get_last_codec_checkpoint_step(os.path.join(data_args.temporary_save_to_disk, split))
                accelerator.wait_for_everyone()
                if start_step > 0:
                    logger.info(f"Resuming {split} from step {start_step}")
                    # efficiently skip the first n batches
                    start_step += 1
                    data_loader = skip_first_batches(data_loader, start_step)

                all_generated_labels = []
                all_lens = []
                if start_step < total_inference_steps:
                    for i, batch in enumerate(tqdm(data_loader, disable=not accelerator.is_local_main_process)):
                        cur_step = start_step + i
                        generate_labels = apply_audio_decoder(batch)
                        generate_labels = accelerator.pad_across_processes(generate_labels, dim=1, pad_index=0)
                        generate_labels = accelerator.gather_for_metrics(generate_labels)

                        if accelerator.is_main_process:
                            lab = generate_labels["labels"].cpu().transpose(1, 2).to(torch.int16)
                            rat = generate_labels["ratio"].cpu().squeeze(1)
                            lens = generate_labels["len_audio"].cpu().squeeze(1)
                            lab = [l[:, : int(ratio * length)] for (l, ratio, length) in zip(lab, rat, lens)]

                            all_generated_labels.extend(lab)
                            all_lens.extend(lens)

                            if ((cur_step + 1) % data_args.save_codec_steps == 0) or (
                                    cur_step == total_inference_steps - 1
                            ):
                                tmp_labels = Dataset.from_dict(
                                    {"labels": all_generated_labels, "target_length": all_lens})
                                tmp_labels = tmp_labels.map(
                                    postprocess_dataset,
                                    num_proc=data_args.preprocessing_num_workers,
                                    # this one is resource consuming if many processor.
                                    input_columns=["labels"],
                                    desc="Postprocessing labeling",
                                )
                                save_codec_checkpoint(
                                    os.path.join(data_args.temporary_save_to_disk, split), tmp_labels, cur_step
                                )
                                all_generated_labels = []
                                all_lens = []

                    accelerator.wait_for_everyone()

                if accelerator.is_main_process and len(all_generated_labels) > 0:
                    tmp_labels = Dataset.from_dict({"labels": all_generated_labels, "target_length": all_lens})
                    tmp_labels = tmp_labels.map(
                        postprocess_dataset,
                        num_proc=data_args.preprocessing_num_workers,
                        # this one is resource consuming if many processor.
                        input_columns=["labels"],
                        desc="Postprocessing labeling",
                    )
                    save_codec_checkpoint(os.path.join(data_args.temporary_save_to_disk, split), tmp_labels, cur_step)
                    all_generated_labels = []
                    all_lens = []
                accelerator.wait_for_everyone()

                del all_generated_labels
                accelerator.wait_for_everyone()

                with accelerator.local_main_process_first():
                    tmp_labels = load_all_codec_checkpoints(
                        os.path.join(data_args.temporary_save_to_disk, split)).select(
                        range(len(vectorized_datasets[split]))
                    )
                    logger.info(f"Concatenating {split}: {tmp_labels} with {vectorized_datasets[split]}")
                    vectorized_datasets[split] = concatenate_datasets([vectorized_datasets[split], tmp_labels], axis=1)
            else:
                logger.info(f"Skipping audio encoding for {split} split (no audio column found)")
                # For eval dataset without audio, we need to add dummy labels and target_length
                # so the dataset structure matches what's expected downstream
                with accelerator.local_main_process_first():
                    def add_dummy_audio_fields(example):
                        # Add dummy labels and target_length for compatibility
                        # Use a target_length that's within the valid range to avoid filtering
                        dummy_duration = (data_args.min_duration_in_seconds + data_args.max_duration_in_seconds) / 2
                        dummy_target_length = int(dummy_duration * sampling_rate)

                        # Create dummy labels with appropriate length
                        # The actual values don't matter since we're generating audio
                        dummy_label_length = dummy_target_length // 320  # Approximate codec downsampling
                        example["labels"] = [[audio_encoder_bos_token_id] * num_codebooks for _ in
                                             range(dummy_label_length)]
                        example["target_length"] = dummy_target_length
                        return example

                    vectorized_datasets[split] = vectorized_datasets[split].map(
                        add_dummy_audio_fields,
                        num_proc=data_args.preprocessing_num_workers,
                        desc=f"Adding dummy audio fields to {split}",
                    )

        accelerator.free_memory()
        if 'generate_labels' in locals():
            del generate_labels
            del all_lens
            # Hardy: The modified section B is to here.

        with accelerator.local_main_process_first():
            # NOTE: filtering is done at the end because in the `datasets` library, caching audio files is done after most operations
            # caching audio files is time and disk-space consuming, so we want to avoid it at all costs, especially for large (>1Kh) audio datasets.
            # That's also why we avoid to concat the processed datasets (vectorized_datasets) with the audio column present in raw_datasets.

            def is_audio_in_length_range(length):
                return length > min_target_length and length < max_target_length

            # filter data that is shorter than min_target_length
            vectorized_datasets = vectorized_datasets.filter(
                is_audio_in_length_range,
                num_proc=num_workers,
                input_columns=["target_length"],
            )

        if description_column_name is not None and data_args.max_description_token_length is not None:
            with accelerator.local_main_process_first():
                # filter description that is shorter than max_text_length
                vectorized_datasets = vectorized_datasets.filter(
                    lambda x: len(x) < data_args.max_description_token_length,
                    num_proc=num_workers,
                    input_columns=["input_ids"],
                )

        if data_args.max_prompt_token_length is not None:
            with accelerator.local_main_process_first():
                # filter description that is shorter than max_text_length
                vectorized_datasets = vectorized_datasets.filter(
                    lambda x: len(x) < data_args.max_prompt_token_length,
                    num_proc=num_workers,
                    input_columns=["prompt_input_ids"],
                )

    if data_args.save_to_disk is not None and not dataset_was_precomputed:
        if accelerator.is_main_process:
            vectorized_datasets.save_to_disk(
                data_args.save_to_disk,
                num_proc=min(data_args.preprocessing_num_workers, len(vectorized_datasets["eval"]) - 1),
            )
        accelerator.wait_for_everyone()
        logger.info(f"Dataset saved at {data_args.save_to_disk}")

    audio_max_length = None
    if padding == "max_length":
        audio_max_length = max(vectorized_datasets["train"]["target_length"])
        with accelerator.local_main_process_first():
            max_sample = vectorized_datasets["train"].filter(
                lambda x: x == audio_max_length,
                num_proc=num_workers,
                input_columns=["target_length"],
            )
        audio_max_length = max([len(l[0]) for l in max_sample["labels"]])

    if description_column_name is not None and data_args.max_description_token_length is not None:
        with accelerator.local_main_process_first():
            # filter description that is shorter than max_text_length
            vectorized_datasets = vectorized_datasets.filter(
                lambda x: len(x) < data_args.max_description_token_length,
                num_proc=num_workers,
                input_columns=["input_ids"],
            )

    if data_args.max_prompt_token_length is not None:
        with accelerator.local_main_process_first():
            # filter description that is shorter than max_text_length
            vectorized_datasets = vectorized_datasets.filter(
                lambda x: len(x) < data_args.max_prompt_token_length,
                num_proc=num_workers,
                input_columns=["prompt_input_ids"],
            )

    if training_args.group_by_length:
        # apply a simple heuristic to take into account audio and text lengths
        def add_target_lengths(target_length, prompt, description):
            return {"target_length": target_length + len(prompt) + len(description)}

        with accelerator.local_main_process_first():
            vectorized_datasets = vectorized_datasets.map(
                add_target_lengths,
                num_proc=num_workers,
                input_columns=["target_length", "prompt_input_ids", "input_ids"],
            )

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with ``args.preprocessing_only`` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step ``args.preprocessing_only`` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only and data_args.save_to_disk is None:
        raise ValueError(
            "`preprocessing_only=True` but `save_to_disk` is not set. The latter should indicates where to save the dataset locally."
        )
    elif data_args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files save at {data_args.save_to_disk}")
        return

    # 6. Next, we can prepare the training.
    # Hardy: Discard the original function of compute_metrics

    # # Let's use word CLAP similary and WER metrics as our evaluation metrics,
    # def compute_metrics(
    #     audios,
    #     descriptions,
    #     prompts,
    #     device="cpu",
    #     compute_clap_similarity_metric=False,
    #     compute_noise_level_metric=False,
    #     noise_level_to_compute_clean_wer=None,
    # ):
    #     results = {}
    #     input_ids = descriptions
    #     texts = description_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    #     prompts = prompt_tokenizer.batch_decode(prompts, skip_special_tokens=True)
    #     audios = [a.float().cpu().numpy() for a in audios]
    #
    #     if compute_clap_similarity_metric:
    #         clap_score = clap_similarity(
    #             model_args.clap_model_name_or_path, texts, audios, device, input_sampling_rate=sampling_rate
    #         )
    #         results["clap"] = clap_score
    #
    #     si_sdr_measures = None
    #     if compute_noise_level_metric:
    #         si_sdr_measures = si_sdr(audios, device, input_sampling_rate=sampling_rate)
    #
    #     word_error, transcriptions, clean_word_error, noisy_word_error, percent_clean_samples = wer(
    #         model_args.asr_model_name_or_path,
    #         prompts,
    #         audios,
    #         device,
    #         training_args.per_device_eval_batch_size,
    #         sampling_rate,
    #         noise_level_to_compute_clean_wer,
    #         si_sdr_measures,
    #     )
    #     results["wer"] = word_error
    #     if clean_word_error is not None:
    #         results["clean_wer"] = clean_word_error
    #         results["noisy_word_error"] = noisy_word_error
    #         results["percent_clean_samples"] = percent_clean_samples
    #
    #     return results, texts, prompts, audios, transcriptions, si_sdr_measures
    # Hardy: The following is the new compute_metrics including ser, clap, and wer
    def compute_metrics(
            audios,
            descriptions,
            prompts,
            emotion_labels=None,
            device="cpu",
            compute_clap_similarity_metric=False,
            compute_noise_level_metric=False,
            compute_ser_metric=False,
            noise_level_to_compute_clean_wer=None,
    ):
        results = {}
        input_ids = descriptions
        texts = description_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        prompts = prompt_tokenizer.batch_decode(prompts, skip_special_tokens=True)
        audios = [a.float().cpu().numpy() for a in audios]

        if compute_clap_similarity_metric:
            clap_score = clap_similarity(
                model_args.clap_model_name_or_path, texts, audios, device, input_sampling_rate=sampling_rate
            )
            results["clap"] = clap_score

        si_sdr_measures = None
        if compute_noise_level_metric:
            si_sdr_measures = si_sdr(audios, device, input_sampling_rate=sampling_rate)

        ser_results = None
        if compute_ser_metric and emotion_labels is not None:
            ser_results = speech_emotion_recognition(
                model_args.ser_model_name_or_path,
                audios,
                emotion_labels,
                device,
                training_args.per_device_eval_batch_size,
                sampling_rate,
            )
            results["ser_accuracy"] = ser_results["ser_accuracy"]
            results["ser_avg_emotion_score"] = ser_results["ser_avg_emotion_score"]

            # Add per-emotion accuracy to results
            if "ser_per_emotion_accuracy" in ser_results:
                for emotion, accuracy in ser_results["ser_per_emotion_accuracy"].items():
                    results[f"ser_accuracy_{emotion}"] = accuracy * 100

        word_error, transcriptions, clean_word_error, noisy_word_error, percent_clean_samples = wer(
            model_args.asr_model_name_or_path,
            prompts,
            audios,
            device,
            training_args.per_device_eval_batch_size,
            sampling_rate,
            noise_level_to_compute_clean_wer,
            si_sdr_measures,
        )
        results["wer"] = word_error
        if clean_word_error is not None:
            results["clean_wer"] = clean_word_error
            results["noisy_word_error"] = noisy_word_error
            results["percent_clean_samples"] = percent_clean_samples

        return results, texts, prompts, audios, transcriptions, si_sdr_measures, ser_results

    # Define Training Schedule
    # Store some constants
    per_device_train_batch_size = int(training_args.per_device_train_batch_size)
    train_batch_size = per_device_train_batch_size * accelerator.num_processes
    gradient_accumulation_steps = int(training_args.gradient_accumulation_steps)
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)

    if training_args.max_steps < 0:
        num_epochs = int(training_args.num_train_epochs)
        steps_per_epoch = len(vectorized_datasets["train"]) // (train_batch_size * gradient_accumulation_steps)
        total_train_steps = steps_per_epoch * num_epochs
    elif training_args.max_steps > 0:
        logger.info("max_steps is given, it will override any value given in num_train_epochs")
        total_train_steps = int(training_args.max_steps)
        # Setting a very large number of epochs so we go as many times as necessary over the iterator.
        num_epochs = sys.maxsize
        steps_per_epoch = total_train_steps

    if training_args.eval_steps is None:
        logger.info(f"eval_steps is not set, evaluating at the end of each epoch")
        eval_steps = steps_per_epoch
    else:
        eval_steps = training_args.eval_steps
        
    if training_args.eval_generation_steps is None:
        eval_generation_steps = eval_steps
    else:
        eval_generation_steps = training_args.eval_generation_steps

    # T5 doesn't support fp16
    autocast_kwargs = AutocastKwargs(enabled=(mixed_precision != "fp16"))

    # Define optimizer, LR scheduler, collator
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )

    # LR scheduler gets stepped by `num_processes` each time -> account for this in warmup / total steps
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.get_warmup_steps(total_train_steps) * accelerator.num_processes,
        num_training_steps=total_train_steps * accelerator.num_processes,
    )

    # Instantiate custom data collator
    data_collator = DataCollatorParlerTTSWithPadding(
        prompt_tokenizer=prompt_tokenizer,
        description_tokenizer=description_tokenizer,
        pad_to_multiple_of=data_args.pad_to_multiple_of,
        padding=padding,
        prompt_max_length=data_args.max_prompt_token_length,
        description_max_length=data_args.max_description_token_length,
        audio_max_length=audio_max_length,
    )

    # Prepare everything with accelerate
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    num_examples = total_train_steps * train_batch_size * gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_examples}")
    logger.info("  Instantaneous batch size per device =" f" {per_device_train_batch_size}")
    logger.info("  Gradient accumulation steps =" f" {gradient_accumulation_steps}")
    logger.info(
        f"  Total train batch size (w. parallel & distributed) = {train_batch_size * gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {total_train_steps}")

    # ======================== Training ================================
    train_time = 0
    train_start = time.time()
    steps_trained_progress_bar = tqdm(
        range(total_train_steps), desc="Train steps ... ", position=0, disable=not accelerator.is_local_main_process
    )
    continue_training = True
    epochs_trained = 0
    cur_step = 0

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if accelerator.is_main_process:
        if training_args.push_to_hub:
            api = HfApi(token=training_args.hub_token)

            # Create repo (repo_name from args or inferred)
            repo_name = training_args.hub_model_id
            if repo_name is None:
                repo_name = Path(training_args.output_dir).absolute().name
            repo_id = api.create_repo(repo_name, exist_ok=True).repo_id

            with open(os.path.join(training_args.output_dir, ".gitignore"), "w+") as gitignore:
                if "wandb" not in gitignore:
                    gitignore.write("wandb\n")
        elif training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Now save everything to be able to create a single processor later
    # make sure all processes wait until data is saved
    # only the main process saves them
    if accelerator.is_main_process:
        # save feature extractor, tokenizer and config
        if (
            model_args.prompt_tokenizer_name is None
            and model_args.description_tokenizer_name
            or (model_args.prompt_tokenizer_name == model_args.description_tokenizer_name)
        ):
            prompt_tokenizer.save_pretrained(training_args.output_dir)
        else:
            logger.warning(
                f"Prompt tokenizer ('{model_args.prompt_tokenizer_name}') and description tokenizer ('{model_args.description_tokenizer_name}') are not the same. Saving only the prompt tokenizer."
            )
            prompt_tokenizer.save_pretrained(training_args.output_dir)

        feature_extractor.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)
    accelerator.wait_for_everyone()

    if checkpoint is not None:
        accelerator.load_state(checkpoint)
        # Find num steps and epoch from saved state string pattern
        pattern = r"checkpoint-(\d+)-epoch-(\d+)"
        match = re.search(pattern, checkpoint)
        cur_step = int(match.group(1))
        epochs_trained = int(match.group(2))

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {cur_step}")

        steps_trained_progress_bar.update(cur_step)

        for epoch in range(0, epochs_trained):
            with accelerator.local_main_process_first():
                vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(training_args.seed)

        if training_args.max_steps < 0:
            # we know exactly the number of steps per epoch, so can skip through the required number of batches
            resume_step = (cur_step - epochs_trained * steps_per_epoch) * gradient_accumulation_steps
        else:
            # Currently we don't know how many steps we've taken in the current epoch
            # So we just shuffle the dataset one extra time and start from a fresh epoch
            # This is "good enough" for our purposes but not fully correct
            resume_step = None
            with accelerator.local_main_process_first():
                vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(training_args.seed)
    else:
        resume_step = None

    gen_kwargs = {
        "do_sample": model_args.do_sample,
        "temperature": model_args.temperature,
        "max_length": model_args.max_length,
        # Because of the delayed pattern mask, generation might stop earlier because of unexpected behaviour
        # on the first tokens of the codebooks that are delayed.
        # This fix the issue.
        "min_new_tokens": num_codebooks + 1,
    }

    # Define gradient update step fn
    def train_step(
        batch,
        accelerator,
        autocast_kwargs,
        num_items_in_batch,
        gradient_accumulation_steps,
    ):
        if mixed_precision == "fp16":
            # fp16 doesn't work with T5-like models
            with accelerator.autocast(autocast_handler=autocast_kwargs):
                if training_args.parallel_mode.value != "distributed":
                    encoder_outputs = model.text_encoder(
                        input_ids=batch.get("input_ids"), attention_mask=batch.get("attention_mask", None)
                    )
                else:
                    encoder_outputs = model.module.text_encoder(
                        input_ids=batch.get("input_ids"), attention_mask=batch.get("attention_mask", None)
                    )
                # we optionnally project last_hidden_state to avoid recomputing every time
                encoder_hidden_states = encoder_outputs.last_hidden_state
                if (
                    config.text_encoder.hidden_size != config.decoder.hidden_size
                    and config.decoder.cross_attention_hidden_size is None
                ):
                    encoder_hidden_states = (
                        model.enc_to_dec_proj(encoder_hidden_states)
                        if training_args.parallel_mode.value != "distributed"
                        else model.module.enc_to_dec_proj(encoder_hidden_states)
                    )

                if batch.get("attention_mask", None) is not None:
                    encoder_hidden_states = encoder_hidden_states * batch.get("attention_mask", None)[..., None]

                encoder_outputs.last_hidden_state = encoder_hidden_states
                batch["encoder_outputs"] = encoder_outputs

        outputs = model(**batch, loss_reduction="sum")
        # CE (data) loss
        ce_loss = (outputs.loss * gradient_accumulation_steps * accelerator.num_processes) / num_items_in_batch

        metrics = {"loss": ce_loss}
        
        # per CE loss
        per_codebook_losses = outputs.per_codebook_losses
        metrics.update({f"codebook_{i}_loss": ((l  * gradient_accumulation_steps * accelerator.num_processes) / num_items_in_batch) for (i,l) in enumerate(per_codebook_losses)})
        return ce_loss, metrics

    # Define eval fn
    def eval_step(
        batch,
        accelerator,
        autocast_kwargs,
    ):
        # Hardy: I added a debug logging here:
        if batch is None:
            logger.error("ERROR: Received None batch in eval_step")
            return {"loss": torch.tensor(0.0)}

        if not batch:
            logger.error("ERROR: Received empty batch in eval_step")
            return {"loss": torch.tensor(0.0)}

        # Log batch contents for debugging
        logger.debug(f"Batch keys: {batch.keys() if hasattr(batch, 'keys') else 'No keys'}")

        eval_model = model if not training_args.torch_compile else model._orig_mod

        if mixed_precision == "fp16":
            # fp16 doesn't work with T5-like models
            with accelerator.autocast(autocast_handler=autocast_kwargs):
                if training_args.parallel_mode.value != "distributed":
                    encoder_outputs = model.text_encoder(
                        input_ids=batch.get("input_ids"), attention_mask=batch.get("attention_mask", None)
                    )
                else:
                    encoder_outputs = model.module.text_encoder(
                        input_ids=batch.get("input_ids"), attention_mask=batch.get("attention_mask", None)
                    )
                # we optionnally project last_hidden_state to avoid recomputing every time
                encoder_hidden_states = encoder_outputs.last_hidden_state
                if (
                    config.text_encoder.hidden_size != config.decoder.hidden_size
                    and config.decoder.cross_attention_hidden_size is None
                ):
                    encoder_hidden_states = (
                        model.enc_to_dec_proj(encoder_hidden_states)
                        if training_args.parallel_mode.value != "distributed"
                        else model.module.enc_to_dec_proj(encoder_hidden_states)
                    )

                if batch.get("attention_mask", None) is not None:
                    encoder_hidden_states = encoder_hidden_states * batch.get("attention_mask", None)[..., None]

                encoder_outputs.last_hidden_state = encoder_hidden_states
                batch["encoder_outputs"] = encoder_outputs

        with torch.no_grad():
            outputs = eval_model(**batch)
        # CE (data) loss
        ce_loss = outputs.loss
        metrics = {"loss": ce_loss}
        
        # per CE loss
        per_codebook_losses = outputs.per_codebook_losses
        metrics.update({f"codebook_{i}_loss": l for (i,l) in enumerate(per_codebook_losses)})
        return metrics

    def generate_step(batch, accelerator):
        batch.pop("decoder_attention_mask", None)
        eval_model = accelerator.unwrap_model(model, keep_fp32_wrapper=True)
        if training_args.torch_compile:
            # if the model is compiled, we use the original model bc compile is not compatible with .generate
            eval_model = model._orig_mod

        # since we've might have loaded the weights in fp32, we have to autocast to ensure FA2 weights are in half-precision.
        # with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=(attn_implementation=="flash_attention_2"))):
        output_audios = eval_model.generate(**batch, **gen_kwargs)
        output_audios = accelerator.pad_across_processes(output_audios, dim=1, pad_index=0)
        return output_audios

    model.train()

    total_batched_samples = resume_step if resume_step is not None else 0
    for epoch in range(epochs_trained, num_epochs):
        with accelerator.local_main_process_first():
            vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(training_args.seed)
        sampler = None
        if training_args.group_by_length:
            sampler = LengthGroupedSampler(train_batch_size, lengths=vectorized_datasets["train"]["target_length"])
        train_dataloader = DataLoader(
            vectorized_datasets["train"],
            collate_fn=data_collator,
            batch_size=per_device_train_batch_size,
            sampler=sampler,
            shuffle=not training_args.group_by_length,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )
        train_dataloader = accelerator.prepare(train_dataloader)
        if hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(epoch)

        if resume_step is not None:
            # Skip the first N batches in the dataloader when resuming from a checkpoint
            logger.info(f"  Skip first {resume_step} batches")
            train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            resume_step = None
            accelerator.wait_for_everyone()

        # We chunkify the epoch iterator into gradient accumulation steps `n` batches
        train_iterator = iter(train_dataloader)
        num_steps_in_epoch = len(train_dataloader)
        remainder = num_steps_in_epoch % gradient_accumulation_steps
        remainder = remainder if remainder != 0 else gradient_accumulation_steps
        total_updates = math.ceil(num_steps_in_epoch / gradient_accumulation_steps)
        
        update_step = -1
        for _ in range(total_updates):
            update_step += 1
            
            # preload the total batch per step
            batch_samples = []
            num_batches_in_step = gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
            for _ in range(num_batches_in_step):
                batch_samples += [next(train_iterator)]
                
            # get num items in batch - if different than BOS and than -100
            num_items_in_batch = sum([(batch["labels"].ne(audio_encoder_bos_token_id) | batch["labels"].ne(-100) | batch["labels"].ne(audio_encoder_eos_token_id)).sum((0,1))[0] for batch in batch_samples])
            num_items_in_batch = accelerator.gather(num_items_in_batch).sum().item()
            
            # losses = []
            for i,batch in enumerate(batch_samples):
                total_batched_samples += 1
                ctx = model.no_sync if (i < len(batch_samples) - 1 and accelerator.num_processes > 1) else contextlib.nullcontext
                
                with ctx():
                    loss, train_metric = train_step(batch, accelerator, autocast_kwargs, num_items_in_batch, gradient_accumulation_steps)
                    accelerator.backward(loss)
                    # losses.append(loss.detach())
            
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # The accelerator has performed an optimization step behind the scenes
            steps_trained_progress_bar.update(1)
            cur_step += 1

            # losses = accelerator.gather(sum(losses)).sum().item() / (accelerator.num_processes * gradient_accumulation_steps)
            
            if cur_step % training_args.logging_steps == 0:
                steps_trained_progress_bar.write(
                    f"Step... ({cur_step} / {total_train_steps} | Loss:"
                    f" {train_metric['loss']}, Learning Rate:"
                    f" {lr_scheduler.get_last_lr()[0]})"
                )
                train_metric["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                log_metric(
                    accelerator,
                    metrics=train_metric,
                    learning_rate=lr_scheduler.get_last_lr()[0],
                    train_time=train_time + time.time() - train_start,
                    step=cur_step,
                    epoch=epoch,
                    prefix="train",
                )

            # save checkpoint and weights after each save_steps and at the end of training
            if (cur_step % training_args.save_steps == 0) or cur_step == total_train_steps:
                intermediate_dir = os.path.join(training_args.output_dir, f"checkpoint-{cur_step}-epoch-{epoch}")
                # safe_serialization=False to avoid shared tensors saving issue (TODO(YL): it's a temporary fix)
                # https://github.com/huggingface/transformers/issues/27293#issuecomment-1872560074
                accelerator.save_state(output_dir=intermediate_dir, safe_serialization=False)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    rotate_checkpoints(
                        training_args.save_total_limit, output_dir=training_args.output_dir, logger=logger
                    )

                    if cur_step == total_train_steps:
                        # un-wrap student model for save
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(training_args.output_dir)

                    if training_args.push_to_hub:
                        api.upload_folder(
                            repo_id=repo_id,
                            folder_path=training_args.output_dir,
                            commit_message=f"Saving train state of step {cur_step}",
                            run_as_future=True,
                        )
                accelerator.wait_for_everyone()

            if training_args.do_eval and (cur_step % eval_steps == 0 or cur_step == total_train_steps):
                train_time += time.time() - train_start
                # ======================== Evaluating ==============================
                model.eval()
                eval_metrics = []
                eval_preds = []
                eval_descriptions = []
                eval_prompts = []
                eval_start = time.time()

                # release training input batch
                batch = release_memory(batch)

                # Hardy: Check if this is a generation-only evaluation
                # has_audio_for_eval = target_audio_column_name in raw_datasets.get("eval",
                #                                                                   {}).column_names if "eval" in raw_datasets else False
                has_audio_for_eval = False
                if isinstance(vectorized_datasets, dict) and "eval" in vectorized_datasets:
                    eval_split = vectorized_datasets["eval"]
                    if hasattr(eval_split, "column_names"):
                        has_audio_for_eval = target_audio_column_name in eval_split.column_names
                logger.info(f"DEBUG: So the has_audio_for_eval is {has_audio_for_eval}")

                if has_audio_for_eval:
                    # Hardy: This is the original one for loss evaluation, and now we tab it and put it under the if-has_audio_for_eval condition
                    validation_dataloader = DataLoader(
                        vectorized_datasets["eval"],
                        collate_fn=data_collator,
                        batch_size=per_device_eval_batch_size,
                        drop_last=False,
                        num_workers=training_args.eval_dataloader_num_workers,
                        pin_memory=training_args.dataloader_pin_memory,
                    )
                    validation_dataloader = accelerator.prepare(validation_dataloader)

                    # Hardy: Debug (this is a legacy debug):
                    logger.info(f"DEBUG: Validation dataloader info:")
                    logger.info(f"  - Number of batches: {len(validation_dataloader)}")
                    logger.info(f"  - Batch size: {per_device_eval_batch_size}")
                    logger.info(f"  - Total samples: {len(vectorized_datasets['eval'])}")
                    # To inspect the first batch:
                    for i, batch in enumerate(validation_dataloader):
                        if i == 0:  # Only check first batch
                            logger.info(f"DEBUG: First batch keys: {batch.keys()}")
                            for key, value in batch.items():
                                if isinstance(value, torch.Tensor):
                                    logger.info(f"  - {key}: shape {value.shape}")
                                else:
                                    logger.info(
                                        f"  - {key}: type {type(value)}, length {len(value) if hasattr(value, '__len__') else 'N/A'}")
                        break
                    # Hardy: up to here

                    for batch in tqdm(
                        validation_dataloader,
                        desc=f"Evaluating - Inference ...",
                        position=2,
                        disable=not accelerator.is_local_main_process,
                    ):
                        # Model forward
                        eval_metric = eval_step(batch, accelerator, autocast_kwargs)
                        eval_metric = accelerator.gather_for_metrics(eval_metric)
                        eval_metric = {key: val.unsqueeze(0) if val.ndim == 0 else val for (key,val) in eval_metric.items()}
                        eval_metrics.append(eval_metric)
                    # Hardy: to here
                # Hardy: New logic
                else:
                    # Skip loss computation for generation-only evaluation
                    logger.info("Skipping loss computation for generation-only evaluation dataset")
                    # Hardy: Due to the reducer matching issue, I replaced the following line with a line building 1-dimension one.
                    # eval_metrics = [{"loss": torch.tensor(0.0)}]  # Dummy metric
                    device = getattr(accelerator, "device", None)
                    eval_metrics = [{"loss": torch.tensor(0.0, device=device).unsqueeze(0)}]  # shape [1]

                # Hardy: Discard the orginal:
                # if training_args.predict_with_generate and (cur_step % eval_generation_steps == 0 or cur_step == total_train_steps):
                #     validation_dataloader = DataLoader(
                #         vectorized_datasets["eval"],
                #         collate_fn=data_collator,
                #         batch_size=per_device_eval_batch_size,
                #         drop_last=False,
                #         num_workers=training_args.eval_dataloader_num_workers,
                #         pin_memory=training_args.dataloader_pin_memory,
                #     )
                #     validation_dataloader = accelerator.prepare(validation_dataloader)
                #     # generation
                #     for batch in tqdm(
                #         validation_dataloader,
                #         desc=f"Evaluating - Generation ...",
                #         position=2,
                #         disable=not accelerator.is_local_main_process,
                #     ):
                #         generated_audios = generate_step(batch, accelerator)
                #         # Gather all predictions and targets
                #         generated_audios, input_ids, prompts = accelerator.pad_across_processes(
                #             (generated_audios, batch["input_ids"], batch["prompt_input_ids"]), dim=1, pad_index=0
                #         )
                #         generated_audios, input_ids, prompts = accelerator.gather_for_metrics(
                #             (generated_audios, input_ids, prompts)
                #         )
                #         eval_preds.extend(generated_audios.to("cpu"))
                #         eval_descriptions.extend(input_ids.to("cpu"))
                #         eval_prompts.extend(prompts.to("cpu"))

                # Hardy: The following is the new one:
                if training_args.predict_with_generate and (
                        cur_step % eval_generation_steps == 0 or cur_step == total_train_steps):
                    validation_dataloader = DataLoader(
                        vectorized_datasets["eval"],
                        collate_fn=data_collator,
                        batch_size=per_device_eval_batch_size,
                        drop_last=False,
                        num_workers=training_args.eval_dataloader_num_workers,
                        pin_memory=training_args.dataloader_pin_memory,
                    )
                    validation_dataloader = accelerator.prepare(validation_dataloader)

                    # Create directory for saving generated audios with timestamp
                    if data_args.generated_audio_save_dir and accelerator.is_main_process:
                        save_dir = os.path.join(data_args.generated_audio_save_dir,
                                                f"checkpoint-{cur_step}-epoch-{epoch}")
                        os.makedirs(save_dir, exist_ok=True)

                    eval_emotion_labels = []
                    eval_metadata = []

                    # generation
                    for batch_idx, batch in enumerate(tqdm(
                            validation_dataloader,
                            desc=f"Evaluating - Generation ...",
                            position=2,
                            disable=not accelerator.is_local_main_process,
                    )):
                        generated_audios = generate_step(batch, accelerator)
                        # Gather all predictions and targets
                        generated_audios, input_ids, prompts = accelerator.pad_across_processes(
                            (generated_audios, batch["input_ids"], batch["prompt_input_ids"]), dim=1, pad_index=0
                        )
                        generated_audios, input_ids, prompts = accelerator.gather_for_metrics(
                            (generated_audios, input_ids, prompts)
                        )

                        # Gather all metadata if present
                        batch_metadata = {}
                        # Hardy: I cried.
                        # metadata_fields = ["emotion", "gender", "background_noise", "pitch", "rate", "test_category",
                        #                    "number"]
                        metadata_fields = ["style", "gender", "noise", "pitch", "speaking_rate", "test_category",
                                           "number"]
                        for field in metadata_fields:
                            if field in batch:
                                batch_metadata[field] = batch[field]

                        if batch_metadata:
                            eval_metadata.extend([batch_metadata] * len(generated_audios))

                        # Extract emotion labels specifically
                        # Hardy: I cried:
                        # if "emotion" in batch:
                        #     eval_emotion_labels.extend(batch["emotion"])
                        if "style" in batch:
                            eval_emotion_labels.extend(batch["style"])

                        eval_preds.extend(generated_audios.to("cpu"))
                        eval_descriptions.extend(input_ids.to("cpu"))
                        eval_prompts.extend(prompts.to("cpu"))

                        # Save generated audios if specified
                        if data_args.generated_audio_save_dir and accelerator.is_main_process:
                            for i, audio in enumerate(generated_audios.to("cpu")):
                                audio_idx = batch_idx * per_device_eval_batch_size + i
                                # Save as numpy array for efficient loading later
                                np.save(os.path.join(save_dir, f"audio_{audio_idx}.npy"), audio.numpy())

                    # Save metadata if we're saving audios
                    if data_args.generated_audio_save_dir and accelerator.is_main_process:
                        metadata = {
                            "step": cur_step,
                            "epoch": epoch,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "descriptions": [d.tolist() for d in eval_descriptions],
                            "prompts": [p.tolist() for p in eval_prompts],
                            "emotion_labels": eval_emotion_labels if eval_emotion_labels else None,
                            "full_metadata": eval_metadata if eval_metadata else None,
                        }
                        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
                            json.dump(metadata, f, indent=2)

                    # Feasibility test: Run full evaluation after first generation step
                    # Hardy: I updated the if-condition logic -- I hope that feasibility test will be always done
                    if not feasibility_test_done:
                    # if not feasibility_test_done and not training_args.post_training_generation_eval:
                        logger.info("*** Running feasibility test for evaluation models ***")

                        # Hardy: I added a debug logging here
                        # This verifies emotion labels were collected during generation
                        if accelerator.is_main_process:
                            logger.info(f"DEBUG: Collected {len(eval_emotion_labels)} emotion labels for SER evaluation")
                            logger.info(
                                f"DEBUG: compute_ser_metric={training_args.compute_ser_metric}, has_labels={len(eval_emotion_labels) > 0}")

                        if accelerator.is_local_main_process:
                            try:
                                # Test with a small subset
                                test_audios = eval_preds[:min(4, len(eval_preds))]
                                test_descriptions = eval_descriptions[:min(4, len(eval_descriptions))]
                                test_prompts = eval_prompts[:min(4, len(eval_prompts))]
                                test_emotion_labels = eval_emotion_labels[:min(4,
                                                                               len(eval_emotion_labels))] if eval_emotion_labels else None

                                # Test all evaluation functions
                                logger.info("Testing WER...")
                                test_wer, _, _, _, _ = wer(
                                    model_args.asr_model_name_or_path,
                                    prompt_tokenizer.batch_decode(test_prompts, skip_special_tokens=True),
                                    [a.float().cpu().numpy() for a in test_audios],
                                    accelerator.device,
                                    training_args.per_device_eval_batch_size,
                                    sampling_rate,
                                    None,
                                    None,
                                )
                                logger.info(f"WER test successful: {test_wer}")

                                if training_args.compute_clap_similarity_metric:
                                    logger.info("Testing CLAP similarity...")
                                    test_clap = clap_similarity(
                                        model_args.clap_model_name_or_path,
                                        description_tokenizer.batch_decode(test_descriptions, skip_special_tokens=True),
                                        [a.float().cpu().numpy() for a in test_audios],
                                        accelerator.device,
                                        input_sampling_rate=sampling_rate
                                    )
                                    logger.info(f"CLAP test successful: {test_clap}")

                                if training_args.compute_noise_level_metric:
                                    logger.info("Testing SI-SDR...")
                                    test_si_sdr = si_sdr(
                                        [a.float().cpu().numpy() for a in test_audios],
                                        accelerator.device,
                                        input_sampling_rate=sampling_rate
                                    )
                                    logger.info(f"SI-SDR test successful: {test_si_sdr}")

                                if training_args.compute_ser_metric and test_emotion_labels:
                                    logger.info("Testing SER...")
                                    test_ser = speech_emotion_recognition(
                                        model_args.ser_model_name_or_path,
                                        [a.float().cpu().numpy() for a in test_audios],
                                        test_emotion_labels,
                                        accelerator.device,
                                        training_args.per_device_eval_batch_size,
                                        sampling_rate,
                                    )
                                    logger.info(f"SER test successful - Accuracy: {test_ser['ser_accuracy']}")
                                    logger.info(f"Emotion mapping: {test_ser['ser_emotion_mapping']}")
                                    if test_ser.get('ser_unmapped_emotion_stats'):
                                        logger.info(f"Unmapped emotion stats: {test_ser['ser_unmapped_emotion_stats']}")

                                feasibility_test_done = True
                                logger.info("*** Feasibility test completed successfully! ***")

                            except Exception as e:
                                logger.error(f"Feasibility test failed: {str(e)}")
                                raise e

                        accelerator.wait_for_everyone()

                    # Only compute metrics if not using post-training evaluation
                    if not training_args.post_training_generation_eval:
                        # Compute metrics now
                        if accelerator.is_local_main_process:
                            (
                                metric_values,
                                pred_descriptions,
                                pred_prompts,
                                audios_np,
                                transcriptions,
                                si_sdr_measures,
                                ser_results,
                            ) = compute_metrics(
                                eval_preds,
                                eval_descriptions,
                                eval_prompts,
                                eval_emotion_labels if eval_emotion_labels else None,
                                accelerator.device,
                                training_args.compute_clap_similarity_metric,
                                training_args.compute_noise_level_metric,
                                training_args.compute_ser_metric and len(eval_emotion_labels) > 0,
                                training_args.noise_level_to_compute_clean_wer,
                            )
                            eval_metrics.update(metric_values)
                            metrics_desc = " ".join([f"Eval {key}: {value} |" for key, value in metric_values.items()])

                            # Log per-emotion accuracy if available
                            if isinstance(ser_results, dict) and "ser_per_emotion_accuracy" in ser_results:
                                for emotion, accuracy in ser_results["ser_per_emotion_accuracy"].items():
                                    eval_metrics[f"ser_accuracy_{emotion}"] = accuracy * 100

                            if "wandb" in training_args.report_to:
                                log_pred(
                                    accelerator,
                                    pred_descriptions,
                                    pred_prompts,
                                    transcriptions,
                                    audios_np,
                                    si_sdr_measures,
                                    sampling_rate=sampling_rate,
                                    step=cur_step,
                                    prefix="eval",
                                )
                        accelerator.wait_for_everyone()

                eval_time = time.time() - eval_start
                # normalize eval metrics
                eval_metrics = {
                    key: torch.mean(torch.cat([d[key] for d in eval_metrics])).to("cpu") for key in eval_metrics[0]
                }

                # compute metrics
                metrics_desc = ""
                if training_args.predict_with_generate and (cur_step % eval_generation_steps == 0 or cur_step == total_train_steps):
                    if accelerator.is_local_main_process:
                        (
                            metric_values,
                            pred_descriptions,
                            pred_prompts,
                            audios,
                            transcriptions,
                            si_sdr_measures,
                        ) = compute_metrics(
                            eval_preds,
                            eval_descriptions,
                            eval_prompts,
                            accelerator.device,
                            training_args.compute_clap_similarity_metric,
                            training_args.compute_noise_level_metric,
                            training_args.noise_level_to_compute_clean_wer,
                        )
                        eval_metrics.update(metric_values)
                        metrics_desc = " ".join([f"Eval {key}: {value} |" for key, value in metric_values.items()])
                        if "wandb" in training_args.report_to:
                            log_pred(
                                accelerator,
                                pred_descriptions,
                                pred_prompts,
                                transcriptions,
                                audios,
                                si_sdr_measures,
                                sampling_rate=sampling_rate,
                                step=cur_step,
                                prefix="eval",
                            )
                    accelerator.wait_for_everyone()

                # Print metrics and update progress bar
                if accelerator.is_local_main_process:
                    steps_trained_progress_bar.write(
                        f"Eval results for step ({cur_step} / {total_train_steps} | Eval Loss: {eval_metrics['loss']} |"
                        f" {metrics_desc})"
                    )

                log_metric(
                    accelerator,
                    metrics=eval_metrics,
                    train_time=eval_time,
                    step=cur_step,
                    epoch=epoch,
                    prefix="eval",
                )

                # release eval batch and relax metrics
                eval_metrics, eval_preds, eval_descriptions, eval_prompts, batch, eval_metric = release_memory(
                    eval_metrics, eval_preds, eval_descriptions, eval_prompts, batch, eval_metric
                )
                if training_args.predict_with_generate and (cur_step % eval_generation_steps == 0 or cur_step == total_train_steps):
                    generated_audios, input_ids, prompts = release_memory(generated_audios, input_ids, prompts)

                # train mode
                model.train()

                # flush the train metrics
                train_start = time.time()

            # break condition
            if cur_step == total_train_steps:
                continue_training = False
                break

        if not continue_training:
            break

    accelerator.end_training()

    # Hardy: Here I added a hugh section for the post-training evaluation.
    # Post-training evaluation
    if training_args.do_eval and training_args.post_training_generation_eval and data_args.generated_audio_save_dir:
        logger.info("*** Running post-training generation evaluation ***")

        # Find all saved checkpoints
        checkpoint_dirs = sorted([
            d for d in os.listdir(data_args.generated_audio_save_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(data_args.generated_audio_save_dir, d))
        ], key=lambda x: int(x.split("-")[1]))

        all_metrics = {}
        emotion_progress_tracking = {}  # Track emotion scores across checkpoints

        for checkpoint_dir in tqdm(checkpoint_dirs, desc="Evaluating checkpoints"):
            checkpoint_path = os.path.join(data_args.generated_audio_save_dir, checkpoint_dir)

            # Extract step and epoch from directory name
            parts = checkpoint_dir.split("-")
            step = int(parts[1])
            epoch = int(parts[3]) if len(parts) > 3 else 0

            # Load metadata
            with open(os.path.join(checkpoint_path, "metadata.json"), "r") as f:
                metadata = json.load(f)

            # Load generated audios
            audio_files = sorted([f for f in os.listdir(checkpoint_path) if f.endswith(".npy")])
            audios = []
            for audio_file in audio_files:
                audio = np.load(os.path.join(checkpoint_path, audio_file))
                audios.append(torch.from_numpy(audio))

            # Reconstruct descriptions and prompts
            eval_descriptions = [torch.tensor(d) for d in metadata["descriptions"]]
            eval_prompts = [torch.tensor(p) for p in metadata["prompts"]]
            emotion_labels = metadata.get("emotion_labels", None)
            full_metadata = metadata.get("full_metadata", None)

            # Compute all metrics
            if accelerator.is_local_main_process:
                try:
                    (
                        metric_values,
                        pred_descriptions,
                        pred_prompts,
                        audios_np,
                        transcriptions,
                        si_sdr_measures,
                        ser_results,
                    ) = compute_metrics(
                        audios,
                        eval_descriptions,
                        eval_prompts,
                        emotion_labels,
                        accelerator.device,
                        training_args.compute_clap_similarity_metric,
                        training_args.compute_noise_level_metric,
                        training_args.compute_ser_metric and emotion_labels is not None,
                        training_args.noise_level_to_compute_clean_wer,
                    )

                    # Store checkpoint info
                    checkpoint_info = {
                        "step": step,
                        "epoch": epoch,
                        "timestamp": metadata.get("timestamp", ""),
                    }

                    # Store metrics for this checkpoint
                    all_metrics[f"checkpoint_{step}"] = {
                        **checkpoint_info,
                        **metric_values,
                    }

                    # Track emotion progress if SER was computed
                    if training_args.compute_ser_metric and emotion_labels and isinstance(ser_results, dict):
                        # Add per-emotion accuracy
                        if "ser_per_emotion_accuracy" in ser_results:
                            for emotion, accuracy in ser_results["ser_per_emotion_accuracy"].items():
                                all_metrics[f"checkpoint_{step}"][f"ser_accuracy_{emotion}"] = accuracy * 100

                                # Track progress
                                if emotion not in emotion_progress_tracking:
                                    emotion_progress_tracking[emotion] = []
                                emotion_progress_tracking[emotion].append({
                                    "step": step,
                                    "epoch": epoch,
                                    "accuracy": accuracy * 100,
                                    "timestamp": metadata.get("timestamp", "")
                                })

                        # Add unmapped emotion stats if available
                        if "ser_unmapped_emotion_stats" in ser_results:
                            all_metrics[f"checkpoint_{step}"]["ser_unmapped_stats"] = ser_results[
                                "ser_unmapped_emotion_stats"]

                    # Log to wandb if enabled
                    if "wandb" in training_args.report_to:
                        wandb_metrics = {f"post_eval/{k}": v for k, v in metric_values.items() if
                                         not isinstance(v, (dict, list))}
                        accelerator.log(wandb_metrics, step=step)

                        # Log per-emotion accuracy
                        if training_args.compute_ser_metric and emotion_labels:
                            for emotion, accuracy_list in emotion_progress_tracking.items():
                                if accuracy_list and accuracy_list[-1]["step"] == step:
                                    accelerator.log(
                                        {f"post_eval/ser_accuracy_{emotion}": accuracy_list[-1]["accuracy"]}, step=step)

                    logger.info(f"Checkpoint {step} (epoch {epoch}): {metric_values}")

                except Exception as e:
                    logger.error(f"Error evaluating checkpoint {step}: {str(e)}")
                    all_metrics[f"checkpoint_{step}"] = {
                        "step": step,
                        "epoch": epoch,
                        "error": str(e)
                    }

        # Save all metrics and emotion progress tracking
        if accelerator.is_main_process:
            # Save detailed metrics
            metrics_path = os.path.join(data_args.generated_audio_save_dir, "all_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(all_metrics, f, indent=2)
            logger.info(f"Saved all post-training metrics to {metrics_path}")

            # Save emotion progress tracking
            if emotion_progress_tracking:
                emotion_progress_path = os.path.join(data_args.generated_audio_save_dir, "emotion_progress.json")
                with open(emotion_progress_path, "w") as f:
                    json.dump(emotion_progress_tracking, f, indent=2)
                logger.info(f"Saved emotion progress tracking to {emotion_progress_path}")

                # Print emotion progress summary
                logger.info("\n=== Emotion Progress Summary ===")
                for emotion, progress in emotion_progress_tracking.items():
                    if progress:
                        start_acc = progress[0]["accuracy"]
                        end_acc = progress[-1]["accuracy"]
                        improvement = end_acc - start_acc
                        logger.info(f"{emotion}: {start_acc:.2f}% → {end_acc:.2f}% (Δ{improvement:+.2f}%)")

            # Generate a summary report
            summary_path = os.path.join(data_args.generated_audio_save_dir, "evaluation_summary.txt")
            with open(summary_path, "w") as f:
                f.write("Post-Training Evaluation Summary\n")
                f.write("=" * 50 + "\n\n")

                # Overall metrics progression
                f.write("Overall Metrics Progression:\n")
                for checkpoint, metrics in sorted(all_metrics.items(), key=lambda x: x[1].get("step", 0)):
                    if "error" not in metrics:
                        f.write(f"\nCheckpoint: Step {metrics['step']}, Epoch {metrics['epoch']}\n")
                        f.write(f"Timestamp: {metrics.get('timestamp', 'N/A')}\n")
                        for key, value in metrics.items():
                            if key not in ["step", "epoch", "timestamp", "ser_unmapped_stats"] and not isinstance(value,
                                                                                                                  (dict,
                                                                                                                   list)):
                                f.write(f"  {key}: {value:.4f}\n")

                # Emotion-specific progress
                if emotion_progress_tracking:
                    f.write("\n\nEmotion-Specific Progress:\n")
                    f.write("-" * 30 + "\n")
                    for emotion, progress in emotion_progress_tracking.items():
                        f.write(f"\n{emotion}:\n")
                        for point in progress:
                            f.write(f"  Step {point['step']} (Epoch {point['epoch']}): {point['accuracy']:.2f}%\n")

                        # Calculate improvement
                        if len(progress) > 1:
                            improvement = progress[-1]["accuracy"] - progress[0]["accuracy"]
                            f.write(f"  Total Improvement: {improvement:+.2f}%\n")

            logger.info(f"Saved evaluation summary to {summary_path}")

            # Print final summary
            logger.info("\n=== Post-Training Evaluation Complete ===")
            logger.info(f"Evaluated {len(checkpoint_dirs)} checkpoints")
            logger.info(f"Results saved to: {data_args.generated_audio_save_dir}")


if __name__ == "__main__":
    main()
