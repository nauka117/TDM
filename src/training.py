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

import torch
from diffusers import StableDiffusionPipeline, LCMScheduler
from diffusers.utils import make_image_grid
from torchvision.utils import save_image
from .models import generate_new


def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
    from accelerate.logging import get_logger
    logger = get_logger(__name__, log_level="INFO")
    logger.info("Running validation... ")
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        scheduler=LCMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
    args.seed = 42
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    prompt = "A photo of a cat"
    image = pipeline(
        prompt=prompt,
        num_inference_steps=4,
        generator=generator,
        guidance_scale=1.0,
    ).images[0]

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    prompt = "A photo of a cat"
    image_teacher = pipeline(
        prompt=prompt,
        num_inference_steps=28,
        generator=generator,
        guidance_scale=7.0,
    ).images[0]

    make_image_grid([image, image_teacher], 1, 2)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            import numpy as np
            np_images = np.stack([np.asarray(img) for img in [image, image_teacher]])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            if is_wandb_available():
                import wandb
                tracker.log(
                    {
                        "validation": [
                            wandb.Image(image, caption=f"Epoch {epoch} | Student"),
                            wandb.Image(image_teacher, caption=f"Epoch {epoch} | Teacher"),
                        ]
                    }
                )

    del pipeline
    torch.cuda.empty_cache()
    return image


def save_validation_images(unet, noise_scheduler, vae, text_encoder, tokenizer, args, accelerator, 
                          fixed_c, fixed_mask, fixed_noise, fixed_T, global_step, output_dir):
    """Save validation images during training"""
    with torch.no_grad():
        fixed_latents = generate_new(unet, noise_scheduler, fixed_noise, fixed_noise, fixed_c, fixed_mask, 
                                    steps=4, total_steps=args.total_steps, use_opensora=True)
        fixed_latents_1step = generate_new(unet, noise_scheduler, fixed_noise, fixed_noise, fixed_c, fixed_mask, 
                                          steps=1, total_steps=args.total_steps, use_opensora=True)
        
        # Open-Sora: Handle 3D VAE
        images_noise = vae.decode(fixed_latents[:4].to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
        images_fixed1 = vae.decode(fixed_latents_1step[:4].to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0].clamp(-1, 1) * 0.5 + 0.5
        images_4step = vae.decode(fixed_latents.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0].clamp(-1,1)*0.5+0.5
        images_noise = images_noise.clamp(-1, 1) * 0.5 + 0.5
        
        # For video, save first frame as image
        if len(images_fixed1.shape) == 5:  # [B, T, C, H, W]
            images_fixed1 = images_fixed1[:, 0]  # Take first frame
        if len(images_noise.shape) == 5:
            images_noise = images_noise[:, 0]
        if len(images_4step.shape) == 5:
            images_4step = images_4step[:, 0]
        
        if accelerator.is_main_process:
            save_image(images_fixed1, f'./{output_dir}/fixed_1step_{global_step}.jpg', normalize=False, nrow=2)
            save_image(images_noise, f'./{output_dir}/fixed_4step_{global_step}.jpg', normalize=False, nrow=2)
            save_image(images_4step[:4], f'./{output_dir}/4step.jpg', normalize=False, nrow=2)
