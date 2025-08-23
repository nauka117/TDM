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

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import logging
import math
import random
import shutil
from pathlib import Path
from copy import deepcopy
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
from transformers.utils import ContextManagers
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, LCMScheduler, StableDiffusionPipeline, UNet2DConditionModel, AutoencoderTiny, Transformer2DModel, DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

import torch as th
from torch import nn
import math

# Open-Sora imports
from opensora.models.causalvideovae import ae_stride_config, ae_channel_config, ae_wrapper
from opensora.models.text_encoder import get_text_warpper
from opensora.models.diffusion import Diffusion_models, Diffusion_models_class
from opensora.utils.utils import explicit_uniform_sampling

# Import our modules
from .args import parse_args
from .utils import compute_snr, predicted_origin, append_dims, extract_into_tensor, get_module_kohya_state_dict
from .models import generate_new
from .predictor import Predictor
from .training import log_validation, save_validation_images

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def main():
    args = parse_args()
    args.output_dir = args.output_dir + f"_cfg{args.cfg}_totalstep{args.total_steps}"
    if args.use_reg:
        args.output_dir = args.output_dir + "-Reg"
    if args.use_huber:
        args.output_dir = args.output_dir + "-Huber"
    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        split_batches=True,
        kwargs_handlers=[diffusers.utils.DeprecationWarning()],
    )

    # Make one log on every process with the configuration for debugging.
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_token is None:
                raise ValueError("Need an `hub_token` to push to hub when `--push_to_hub` is passed.")
            create_repo(
                repo_id=args.hub_model_id or args.output_dir,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_name_1, subfolder="tokenizer", use_fast=False)

    # Open-Sora: Initialize VAE
    ae = ae_wrapper[args.ae](args.ae_path)
    
    # Open-Sora: Initialize text encoders
    text_enc_1 = get_text_warpper(args.text_encoder_name_1)(args.text_encoder_name_1)
    text_enc_2 = None
    if args.text_encoder_name_2:
        text_enc_2 = get_text_warpper(args.text_encoder_name_2)(args.text_encoder_name_2)

    # Open-Sora: Get VAE configuration
    latent_size = args.max_height // 8  # VAE downsampling factor
    latent_size_t = args.num_frames // ae_stride_config[args.ae][0]  # Temporal downsampling
    channels = ae_channel_config[args.ae]

    # Open-Sora: Initialize diffusion model (DiT)
    if args.model in Diffusion_models_class:
        model = Diffusion_models_class[args.model].from_pretrained(
            args.ae_path,
            subfolder="model",
            cache_dir=args.cache_dir
        )
    else:
        # Fallback to direct instantiation
        model = Diffusion_models[args.model](
            in_channels=channels,
            out_channels=channels,
            sample_size_h=latent_size,
            sample_size_w=latent_size,
            sample_size_t=latent_size_t,
            interpolation_scale_h=args.interpolation_scale_h,
            interpolation_scale_w=args.interpolation_scale_w,
            interpolation_scale_t=args.interpolation_scale_t,
            sparse1d=args.sparse1d, 
            sparse_n=args.sparse_n, 
            skip_connection=args.skip_connection, 
        )
        
        # Load pretrained weights if specified
        if args.pretrained:
            model_state_dict = model.state_dict()
            print(f'Load from {args.pretrained}')
            if args.pretrained.endswith('.safetensors'):  
                from safetensors.torch import load_file as safe_load
                pretrained_checkpoint = safe_load(args.pretrained, device="cpu")
            else:
                pretrained_checkpoint = torch.load(args.pretrained, map_location="cpu")
            
            pretrained_keys = set(list(pretrained_checkpoint.keys()))
            model_keys = set(list(model_state_dict.keys()))
            missing_keys = model_keys - pretrained_keys
            unexpected_keys = pretrained_keys - model_keys
            
            if len(missing_keys) > 0:
                print(f"Missing keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                print(f"Unexpected keys: {unexpected_keys}")
            
            model.load_state_dict(pretrained_checkpoint, strict=False)
        
        # Create fake model for TDM
        model_fake = Diffusion_models[args.model](
            in_channels=channels,
            out_channels=channels,
            sample_size_h=latent_size,
            sample_size_w=latent_size,
            sample_size_t=latent_size_t,
            interpolation_scale_h=args.interpolation_scale_h,
            interpolation_scale_w=args.interpolation_scale_w,
            interpolation_scale_t=args.interpolation_scale_t,
            sparse1d=args.sparse1d, 
            sparse_n=args.sparse_n, 
            skip_connection=args.skip_connection, 
        )
        
        # Enable optimizations
        if hasattr(model, 'enable_xformers_memory_efficient_attention'):
            model.enable_xformers_memory_efficient_attention()
        if hasattr(model_fake, 'enable_xformers_memory_efficient_attention'):
            model_fake.enable_xformers_memory_efficient_attention()
        
        # Set trainable states
        ae.requires_grad_(False)
        text_enc_1.requires_grad_(False)
        if text_enc_2 is not None:
            text_enc_2.requires_grad_(False)
        model_fake.requires_grad_(False)
        model.train()
        model_fake.train()
        
        # Store for later use
        vae = ae
        text_encoder = [text_enc_1, text_enc_2] if text_enc_2 is not None else [text_enc_1]
        unet = model
        unet_fake = model_fake

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters())

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        unet_fake.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be automatically downloaded from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            data_files=args.train_data_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets and DataLoaders creation.
    augmentations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}

    if args.train_data_dir:
        dataset["train"].set_transform(transform_images)

    def collate_fn(examples):
        images = torch.stack([example["input"] for example in examples])
        return images

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"],
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        unet, optimizer, lr_scheduler, train_dataloader
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping the weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text encoders and VAE to GPU and cast to weight_dtype
    ae.to(accelerator.device, dtype=weight_dtype)
    text_enc_1.to(accelerator.device, dtype=weight_dtype)
    if text_enc_2 is not None:
        text_enc_2.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset['train'])}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on the main process.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    # Open-Sora: Initialize teacher model (SD) for TDM
    if args.use_sd_teacher:
        # Load Stable Diffusion teacher model
        from diffusers import StableDiffusionPipeline
        teacher_pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=weight_dtype
        ).to(accelerator.device)
        unet_sd = teacher_pipeline.unet
        unet_sd.requires_grad_(False)
        unet_sd.eval()
    else:
        unet_sd = None

    # Open-Sora: Prepare unconditional embeddings for CFG
    if args.cfg > 1:
        uncond_input = tokenizer(
            [""] * args.train_batch_size,
            return_tensors="pt",
            padding="max_length", max_length=120
        ).to(accelerator.device)
        uncond_attention_mask = uncond_input.attention_mask.to(weight_dtype)
        uncond_prompt_embeds = text_enc_1(uncond_input.input_ids, return_dict=False, attention_mask=uncond_input.attention_mask)[0]
        fixed_input = tokenizer(
            ["A photo of a cat", "A photo of a dog", "A photo of a panda", "A photo of a pikachu"],
            return_tensors="pt",
            padding="max_length", max_length=120
        ).to(accelerator.device)
        fixed_prompt_embeds = text_enc_1(fixed_input.input_ids, return_dict=False, attention_mask=fixed_input.attention_mask)[0]
        fixed_mask = fixed_input.attention_mask.to(weight_dtype)
        add_cfg = {"uncond_attention_mask": uncond_attention_mask[:fixed_mask.shape[0]], "uncond_prompt_embeds": uncond_prompt_embeds[:fixed_mask.shape[0]], 'cfg': 7.5}

    # Open-Sora: Initialize predictor for TDM
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod).to(accelerator.device).to(weight_dtype)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod).to(accelerator.device).to(weight_dtype)
    
    if unet_sd is not None:
        predictor = Predictor(noise_scheduler, alpha_schedule, sigma_schedule, uncond_prompt_embeds, uncond_attention_mask)

    fixed_noise = None
    fixed_c = None
    total_steps = args.total_steps
    
    # Open-Sora: Main training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0    
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, unet_fake):
                text_ = list(batch[0])
                
                # Open-Sora: 3D tensors [B, C, T, H, W]
                noise = torch.randn([len(text_), channels, latent_size_t, latent_size, latent_size]).to(weight_dtype).to(accelerator.device)
                latents = noise

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
    
                new_noise = torch.randn_like(noise)
                bsz = noise.shape[0]
                # Sample a random timestep for each image
                T = torch.randint(total_steps - 1, total_steps, (bsz,), device=noise.device).long()
                
                # Get the text embedding for conditioning
                with torch.no_grad():
                    # Open-Sora: Handle multiple text encoders
                    input_ids = batch[1].view(batch[1].shape[0],-1)
                    prompt_attention_mask = batch[2].view(batch[1].shape[0],-1)
                    
                    # Process with first text encoder (T5)
                    encoder_hidden_states_1 = text_enc_1(input_ids, return_dict=False, attention_mask=prompt_attention_mask)[0]
                    
                    # Process with second text encoder (CLIP) if available
                    if len(text_encoder) > 1 and text_encoder[1] is not None:
                        encoder_hidden_states_2 = text_encoder[1](input_ids, return_dict=False, attention_mask=prompt_attention_mask)[0]
                        encoder_hidden_states = [encoder_hidden_states_1, encoder_hidden_states_2]
                    else:
                        encoder_hidden_states = encoder_hidden_states_1
                    
                
                with torch.no_grad():
                    imgs_list, noisy_imgs_list = generate_new(unet, noise_scheduler, noise, noise, encoder_hidden_states, prompt_attention_mask, 
                                                             steps=4, return_mid=True, total_steps=args.total_steps, 
                                                             use_opensora=True)
                    noisy_imgs_list.reverse()
                
                fw_t = 240
                # Train the Fake score model
                unet_fake.requires_grad_(True)
                with torch.no_grad():
                    encoder_hidden_states_fake = encoder_hidden_states
                    ind_t = torch.randint(1, 5, (bsz,), device=noise.device).long()
                    if args.use_separate:
                        t_fake = torch.randint(fw_t, fw_t + 200, (bsz,), device=noise.device).long()
                    else:
                        t_fake = torch.randint(fw_t, total_steps, (bsz,), device=noise.device).long()
                    noise_fake = torch.randn_like(noise)
                    latents_fake = predictor.add_noise(noisy_imgs_list[ind_t], noise_fake, ind_t * total_steps // 4, t_fake)
                    
                    # Open-Sora: Prepare model kwargs for DiT
                    model_kwargs = {
                        "encoder_hidden_states": encoder_hidden_states_fake,
                        "attention_mask": None,  # Open-Sora doesn't use this
                        "encoder_attention_mask": prompt_attention_mask,
                    }
                    
                    noise_pred_fake = unet_fake(latents_fake, timestep=t_fake, **model_kwargs, return_dict=False)[0]
                    noise_pred_fake = noise_pred_fake.chunk(2, dim=1)[0]
                    fake_latents = predicted_origin(noise_pred_fake,
                                                    t_fake,
                                                    latents_fake,
                                                    noise_scheduler.config.prediction_type,
                                                    alpha_schedule,
                                                    sigma_schedule, )
                    fake_latents = fake_latents.detach().clone()
                    fake_latents_uncond = fake_latents
                    if args.cfg > 1:
                        noise_pred_fake_uncond = unet_fake(latents_fake, timestep=t_fake, **model_kwargs,
                                                            encoder_hidden_states=uncond_prompt_embeds, encoder_attention_mask=uncond_attention_mask, return_dict=False)[0]
                        noise_pred_fake_uncond = noise_pred_fake_uncond.chunk(2, dim=1)[0]
                        fake_latents_uncond = predicted_origin(noise_pred_fake_uncond,
                                                                t_fake,
                                                                latents_fake,
                                                                noise_scheduler.config.prediction_type,
                                                                alpha_schedule,
                                                                sigma_schedule, )
                        fake_latents_uncond = fake_latents_uncond.detach().clone()

                # Train the Student model
                unet_fake.requires_grad_(False)
                unet.requires_grad_(True)
                with torch.no_grad():
                    if unet_sd is not None:
                        sd_latents, sd_latents_uncond = predictor.predict(unet_sd, fake_latents, t_fake, encoder_hidden_states, prompt_attention_mask, cfg=args.cfg, steps=1, return_double=True)
                        sd_latents = sd_latents.detach().clone()
                        sd_latents_uncond = sd_latents_uncond.detach().clone()
                    else:
                        # Use fake model as teacher if no SD teacher
                        sd_latents = fake_latents
                        sd_latents_uncond = fake_latents_uncond

                # Open-Sora: Predict with student model
                model_kwargs = {
                    "encoder_hidden_states": encoder_hidden_states,
                    "attention_mask": None,
                    "encoder_attention_mask": prompt_attention_mask,
                }
                
                model_latents, _ = predictor.predict(unet, fake_latents, t_fake, encoder_hidden_states, prompt_attention_mask, cfg=args.cfg, steps=1)

                if args.cfg > 1:
                    coop_samples = sd_latents + (args.cfg-1) * (sd_latents - sd_latents_uncond)
                else:
                    coop_samples = sd_latents
                sd_latents_cfg = sd_latents + (args.cfg-1) * (sd_latents - sd_latents_uncond)
                weighting_factor = torch.abs(model_latents.double() - sd_latents_cfg.double() ).mean(dim=[1, 2, 3], keepdim=True).detach()
                if args.use_huber:
                    args.huber_c = 1e-3
                    loss_instruct = torch.mean(
                    (torch.sqrt((model_latents.float() - coop_samples.detach().float()) ** 2 + args.huber_c**2) - args.huber_c) / weighting_factor
                )
                else:
                    loss_instruct = F.mse_loss(model_latents.float(), coop_samples.detach().float(),
                                            reduction='none') / weighting_factor
                    loss_instruct = loss_instruct.mean()
                loss = loss_instruct
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                if global_step % 50 == 0:
                    if fixed_c is None:
                        fixed_c = fixed_prompt_embeds.to(weight_dtype).to(accelerator.device)
                        fixed_mask = fixed_mask.to(weight_dtype).to(accelerator.device)
                        
                        # Open-Sora: 3D tensors [B, C, T, H, W]
                        fixed_noise = torch.randn([4, channels, latent_size_t, latent_size, latent_size]).to(weight_dtype).to(accelerator.device)
                        fixed_T = T
                    save_validation_images(unet, noise_scheduler, vae, text_encoder, tokenizer, args, accelerator, 
                                          fixed_c, fixed_mask, fixed_noise, fixed_T, global_step, args.output_dir)

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            if args.use_ema:
                ema_unet.step(unet.parameters())
            progress_bar.update(1)
            global_step += 1
            accelerator.log({"loss_instruct": loss_instruct, },
                            step=global_step)
            train_loss = 0.0
            train_d_real = 0.0
            train_d_fake = 0.0

            if global_step % (args.checkpointing_steps) == 0:
                if accelerator.is_main_process:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

        logs = {"loss_instruct": loss_instruct.detach().item(), }
        progress_bar.set_postfix(**logs)

        if global_step >= args.max_train_steps:
            break
    accelerator.end_training()


if __name__ == "__main__":
    main()
