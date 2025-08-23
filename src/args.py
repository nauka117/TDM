#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    # Open-Sora specific arguments
    parser.add_argument(
        "--ae",
        type=str,
        default="WFVAEModel_D8_4x8x8",
        help="VAE model configuration for Open-Sora",
    )
    parser.add_argument(
        "--ae_path",
        type=str,
        default="LanguageBind/Open-Sora-Plan-v1.3.0",
        help="Path to VAE model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="OpenSoraT2V_v1_3_93x640x640",
        help="Diffusion model configuration for Open-Sora",
    )
    parser.add_argument(
        "--text_encoder_name_1",
        type=str,
        default="google/t5-v1_1-xl",
        help="First text encoder for Open-Sora",
    )
    parser.add_argument(
        "--text_encoder_name_2",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Second text encoder for Open-Sora (optional)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=93,
        help="Number of frames for video generation",
    )
    parser.add_argument(
        "--max_height",
        type=int,
        default=640,
        help="Maximum height for video generation",
    )
    parser.add_argument(
        "--max_width",
        type=int,
        default=640,
        help="Maximum width for video generation",
    )
    parser.add_argument(
        "--interpolation_scale_h",
        type=float,
        default=1.0,
        help="Horizontal interpolation scale",
    )
    parser.add_argument(
        "--interpolation_scale_w",
        type=float,
        default=1.0,
        help="Vertical interpolation scale",
    )
    parser.add_argument(
        "--interpolation_scale_t",
        type=float,
        default=1.0,
        help="Temporal interpolation scale",
    )
    parser.add_argument(
        "--sparse1d",
        action="store_true",
        help="Use sparse 1D attention",
    )
    parser.add_argument(
        "--sparse_n",
        type=int,
        default=2,
        help="Sparse attention parameter",
    )
    parser.add_argument(
        "--skip_connection",
        action="store_true",
        help="Use skip connections",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for models",
    )
    parser.add_argument(
        "--use_sd_teacher",
        action="store_true",
        help="Use Stable Diffusion as teacher model for TDM",
    )
    
    # Standard training arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the Dataset (from the HuggingFace hub) to train on (could be your own dataset, provided it is formatted to the `imagefolder` format).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as `None` if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data. Folder contents must follow the structure described in https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file must exist to provide the captions for the images. Ignored if `dataset_name` is specified.",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this "
        "value if set.",
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        nargs="+",
        help="A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."
        " Provide either a matching number of `--validation_images`, or a single `--validation_image`"
        " to be used with all prompts.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates. This can be used to resume training via `--resume_from_checkpoint`.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. This can speed up training by ~20% but can reduce"
            " numerical precision. https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"none"` to disable any'
            " logging."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xFormers."
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    
    # TDM specific arguments
    parser.add_argument(
        "--total_steps",
        type=int,
        default=900,
        help="Total number of diffusion steps for TDM",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=4.5,
        help="CFG scale for TDM",
    )
    parser.add_argument(
        "--use_huber",
        action="store_true",
        help="Use Huber loss for TDM",
    )
    parser.add_argument(
        "--use_separate",
        action="store_true",
        help="Use separate noise intervals for TDM",
    )
    parser.add_argument(
        "--use_reg",
        action="store_true",
        help="Use regularization for TDM",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained model weights",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the Dataset (from the HuggingFace hub) to train on (could be your own dataset, provided it is formatted to the `imagefolder` format).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as `None` if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data. Folder contents must follow the structure described in https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file must exist to provide the captions for the images. Ignored if `dataset_name` is specified.",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this "
        "value if set.",
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        nargs="+",
        help="A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."
        " Provide either a matching number of `--validation_images`, or a single `--validation_image`"
        " to be used with all prompts.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates. This can be used to resume training via `--resume_from_checkpoint`.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. This can speed up training by ~20% but can reduce"
            " numerical precision. https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"none"` to disable any'
            " logging."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xFormers."
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    
    # TDM specific arguments
    parser.add_argument(
        "--total_steps",
        type=int,
        default=900,
        help="Total number of diffusion steps for TDM",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=4.5,
        help="CFG scale for TDM",
    )
    parser.add_argument(
        "--use_huber",
        action="store_true",
        help="Use Huber loss for TDM",
    )
    parser.add_argument(
        "--use_separate",
        action="store_true",
        help="Use separate noise intervals for TDM",
    )
    parser.add_argument(
        "--use_reg",
        action="store_true",
        help="Use regularization for TDM",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained model weights",
    )
    
    args = parser.parse_args()
    
    # Set default values for Open-Sora
    if args.pretrained_model_name_or_path is None:
        args.pretrained_model_name_or_path = args.ae_path
    
    return args

        "--lora_dropout",
        type=float,
        default=0.0,
        help="The dropout probability for the LoRA adapter.",
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        help="The bias type for the LoRA adapter.",
    )
    parser.add_argument(
        "--lora_text_encoder_r",
        type=int,
        default=16,
        help="The rank of the LoRA adapter for the text encoder.",
    )
    parser.add_argument(
        "--lora_text_encoder_alpha",
        type=int,
        default=32,
        help="The alpha parameter for the LoRA adapter for the text encoder.",
    )
    parser.add_argument(
        "--lora_text_encoder_dropout",
        type=float,
        default=0.0,
        help="The dropout probability for the LoRA adapter for the text encoder.",
    )
    parser.add_argument(
        "--lora_text_encoder_bias",
        type=str,
        default="none",
        help="The bias type for the LoRA adapter for the text encoder.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for more information see"
            " https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=4.5,
        help="CFG scale for training.",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=900,
        help="Total number of diffusion steps.",
    )
    parser.add_argument(
        "--use_reg",
        action="store_true",
        help="Use regularization.",
    )
    parser.add_argument(
        "--use_huber",
        action="store_true",
        help="Use Huber loss.",
    )
    parser.add_argument(
        "--use_separate",
        action="store_true",
        help="Use separate noise intervals.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        help="Enable tiling for VAE.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained model weights.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log interval for training.",
    )
    parser.add_argument(
        "--offload_ema",
        action="store_true",
        help="Offload EMA model to CPU.",
    )
    parser.add_argument(
        "--rf_scheduler",
        action="store_true",
        help="Use RF scheduler.",
    )
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=0.0,
        help="Noise offset for training.",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="uniform",
        help="Weighting scheme for loss.",
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="Logit mean for timestep sampling.",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="Logit std for timestep sampling.",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.0,
        help="Mode scale for timestep sampling.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        help="Enable tiling for VAE.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained model weights.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log interval for training.",
    )
    parser.add_argument(
        "--offload_ema",
        action="store_true",
        help="Offload EMA model to CPU.",
    )
    parser.add_argument(
        "--rf_scheduler",
        action="store_true",
        help="Use RF scheduler.",
    )
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=0.0,
        help="Noise offset for training.",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="uniform",
        help="Weighting scheme for loss.",
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="Logit mean for timestep sampling.",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="Logit std for timestep sampling.",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.0,
        help="Mode scale for timestep sampling.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args
