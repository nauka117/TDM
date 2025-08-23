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
    """
    Log validation results for Open-Sora models.
    
    Args:
        vae: VAE model (Open-Sora VAE wrapper)
        text_encoder: Text encoder(s) for Open-Sora
        tokenizer: Tokenizer for text encoding
        unet: Diffusion model (DiT for Open-Sora)
        args: Training arguments
        accelerator: Accelerator instance
        weight_dtype: Weight data type
        epoch: Current epoch
    """
    from accelerate.logging import get_logger
    logger = get_logger(__name__, log_level="INFO")
    logger.info("Running validation... ")
    
    # Open-Sora: Create pipeline for validation
    try:
        from opensora.sample.pipeline_opensora import OpenSoraPipeline
        
        # Handle multiple text encoders
        if isinstance(text_encoder, list):
            text_enc_1 = text_encoder[0]
            text_enc_2 = text_encoder[1] if len(text_encoder) > 1 else None
        else:
            text_enc_1 = text_encoder
            text_enc_2 = None
        
        # Create Open-Sora pipeline
        pipeline = OpenSoraPipeline(
            vae=vae,
            text_encoder_1=text_enc_1,
            text_encoder_2=text_enc_2,
            tokenizer=tokenizer,
            scheduler=LCMScheduler(),
            transformer=accelerator.unwrap_model(unet)
        )
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        
        if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
            pipeline.enable_xformers_memory_efficient_attention()
        
        # Set seed for reproducibility
        args.seed = 42
        if args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

        # Run inference with student model (4 steps)
        prompt = "A photo of a cat"
        student_video = pipeline(
            prompt=prompt,
            num_inference_steps=4,
            generator=generator,
            guidance_scale=1.0,
        ).frames[0]

        # Run inference with teacher model (28 steps)
        teacher_video = pipeline(
            prompt=prompt,
            num_inference_steps=28,
            generator=generator,
            guidance_scale=7.0,
        ).frames[0]

        # Create comparison grid
        # Note: For videos, we might want to save individual frames or create a video comparison
        comparison_result = make_image_grid([student_video[0], teacher_video[0]], 1, 2)
        
        # Log to tensorboard/wandb
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                import numpy as np
                # Convert video frames to numpy for logging
                np_student = np.stack([np.asarray(frame) for frame in student_video[:4]])  # First 4 frames
                np_teacher = np.stack([np.asarray(frame) for frame in teacher_video[:4]])
                np_videos = np.concatenate([np_student, np_teacher], axis=0)
                tracker.writer.add_images("validation", np_videos, epoch, dataformats="NHWC")
                
            if tracker.name == "wandb":
                if is_wandb_available():
                    import wandb
                    tracker.log(
                        {
                            "validation": [
                                wandb.Video(student_video, caption=f"Epoch {epoch} | Student (4 steps)"),
                                wandb.Video(teacher_video, caption=f"Epoch {epoch} | Teacher (28 steps)"),
                            ]
                        }
                    )
                    
    except ImportError:
        # Fallback to standard pipeline if Open-Sora pipeline not available
        logger.warning("Open-Sora pipeline not available, using fallback validation")
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=accelerator.unwrap_model(vae),
            text_encoder=accelerator.unwrap_model(text_encoder[0] if isinstance(text_encoder, list) else text_encoder),
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

        if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
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

        comparison_result = make_image_grid([image, image_teacher], 1, 2)

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


def save_validation_images(unet, noise_scheduler, vae, text_encoder, tokenizer, args, accelerator, 
                          fixed_c, fixed_mask, fixed_noise, fixed_T, global_step, output_dir):
    """
    Save validation images/videos for Open-Sora models.
    
    Args:
        unet: Diffusion model (DiT for Open-Sora)
        noise_scheduler: Noise scheduler
        vae: VAE model (Open-Sora VAE wrapper)
        text_encoder: Text encoder(s) for Open-Sora
        tokenizer: Tokenizer for text encoding
        args: Training arguments
        accelerator: Accelerator instance
        fixed_c: Fixed prompt embeddings
        fixed_mask: Fixed attention mask
        fixed_noise: Fixed noise tensor
        fixed_T: Fixed timesteps
        global_step: Current global step
        output_dir: Output directory for saving
    """
    from accelerate.logging import get_logger
    logger = get_logger(__name__, log_level="INFO")
    
    logger.info("Saving validation images/videos...")
    
    try:
        # Open-Sora: Generate validation samples
        with torch.no_grad():
            # Handle multiple text encoders
            if isinstance(text_encoder, list):
                encoder_hidden_states = [fixed_c, None] if len(text_encoder) > 1 else [fixed_c]
            else:
                encoder_hidden_states = fixed_c
            
            # Generate samples using TDM approach
            from .models import generate_new
            generated_latents = generate_new(
                unet, noise_scheduler, fixed_noise, fixed_noise, 
                encoder_hidden_states, fixed_mask, 
                steps=4, return_mid=False, total_steps=args.total_steps, 
                use_opensora=True
            )
            
            # Decode latents to images/videos
            if hasattr(vae, 'decode'):
                # Open-Sora VAE wrapper
                generated_samples = vae.decode(generated_latents)
            else:
                # Standard VAE
                generated_samples = vae.decode(generated_latents / 0.18215)
            
            # Save samples
            for i, sample in enumerate(generated_samples):
                if len(sample.shape) == 4:  # Video: [T, C, H, W]
                    # Save as video frames
                    for t, frame in enumerate(sample):
                        save_image(
                            frame, 
                            f"{output_dir}/validation_step_{global_step}_sample_{i}_frame_{t}.png"
                        )
                else:  # Image: [C, H, W]
                    save_image(
                        sample, 
                        f"{output_dir}/validation_step_{global_step}_sample_{i}.png"
                    )
                    
        logger.info(f"Saved validation samples to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error saving validation samples: {e}")
        # Fallback: save noise as placeholder
        save_image(
            fixed_noise[0, :, 0, :, :],  # First sample, first frame
            f"{output_dir}/validation_step_{global_step}_placeholder.png"
        )
