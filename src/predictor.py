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
from .utils import predicted_origin, extract_into_tensor


class Predictor():
    def __init__(self, noise_scheduler, alpha_schedule, sigma_schedule, uncond_prompt_embeds, uncond_attention_mask):
        super(Predictor, self).__init__()
        self.noise_scheduler = noise_scheduler
        self.alpha_schedule = alpha_schedule
        self.sigma_schedule = sigma_schedule
        self.uncond_prompt_embeds = uncond_prompt_embeds
        self.uncond_attention_mask = uncond_attention_mask
        self.added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

    def predict(self, score_model, noisy_samples, timesteps, encoder_hidden_states, prompt_attention_mask, cfg=None, steps=1,
                return_double=False, timestep_cond=None, use_opensora=False):
        """
        Predict noise or latents using the score model.
        
        Args:
            score_model: The diffusion model (DiT for Open-Sora)
            noisy_samples: Input noisy samples [B, C, T, H, W]
            timesteps: Current timesteps
            encoder_hidden_states: Text embeddings
            prompt_attention_mask: Attention mask for text
            cfg: CFG scale
            steps: Number of prediction steps
            return_double: Whether to return both CFG and non-CFG predictions
            timestep_cond: Additional timestep conditioning
            use_opensora: Whether using Open-Sora architecture
        """
        for _ in range(steps):
            # Open-Sora: Prepare model kwargs for DiT
            if use_opensora:
                model_kwargs = {
                    "encoder_hidden_states": encoder_hidden_states,
                    "attention_mask": None,  # Open-Sora doesn't use this
                    "encoder_attention_mask": prompt_attention_mask,
                }
            else:
                model_kwargs = {
                    "encoder_hidden_states": encoder_hidden_states,
                    "attention_mask": prompt_attention_mask,
                    "added_cond_kwargs": self.added_cond_kwargs,
                }
            
            # Get score prediction
            score_pred = score_model(noisy_samples, timestep=timesteps, **model_kwargs, return_dict=False)[0]
            
            # Handle different output formats
            if hasattr(score_pred, 'chunk'):
                score_pred = score_pred.chunk(2, dim=1)[0]
            
            if cfg is not None:
                # Get unconditional prediction for CFG
                if use_opensora:
                    uncond_model_kwargs = {
                        "encoder_hidden_states": self.uncond_prompt_embeds,
                        "attention_mask": None,
                        "encoder_attention_mask": self.uncond_attention_mask,
                    }
                else:
                    uncond_model_kwargs = {
                        "encoder_hidden_states": self.uncond_prompt_embeds,
                        "attention_mask": self.uncond_attention_mask,
                        "added_cond_kwargs": self.added_cond_kwargs,
                    }
                
                score_uncon_pred = score_model(noisy_samples, timestep=timesteps, **uncond_model_kwargs, return_dict=False)[0]
                
                if hasattr(score_uncon_pred, 'chunk'):
                    score_uncon_pred = score_uncon_pred.chunk(2, dim=1)[0]
                
                # Apply CFG
                score_pred_cfg = score_uncon_pred + cfg * (score_pred - score_uncon_pred)
                
                # Predict latents with CFG
                pred_latents = predicted_origin(
                    score_pred_cfg,
                    timesteps.long(),
                    noisy_samples,
                    self.noise_scheduler.config.prediction_type,
                    self.alpha_schedule,
                    self.sigma_schedule, )
                
                # Predict latents without CFG (for return_double)
                if return_double:
                    pred_latents_nocfg = predicted_origin(
                        score_pred,
                        timesteps.long(),
                        noisy_samples,
                        self.noise_scheduler.config.prediction_type,
                        self.alpha_schedule,
                        self.sigma_schedule, )
                
                # Update timesteps and add noise for next step
                timesteps = timesteps - timesteps // steps
                noisy_samples = self.noise_scheduler.add_noise(pred_latents.detach(), score_pred_cfg, timesteps.long())
                
                if return_double:
                    return score_pred_cfg, pred_latents, pred_latents_nocfg
            else:
                # No CFG - direct prediction
                pred_latents = predicted_origin(
                    score_pred,
                    timesteps,
                    noisy_samples,
                    self.noise_scheduler.config.prediction_type,
                    self.alpha_schedule,
                    self.sigma_schedule, )
                
                # Update timesteps and add noise for next step
                timesteps = timesteps - timesteps // steps
                noisy_samples = self.noise_scheduler.add_noise(pred_latents.detach(), score_pred, timesteps)
        
        if cfg is not None:
            return score_pred_cfg, pred_latents
        else:
            return score_pred, pred_latents
        
    def add_noise(self, samples, noise, t1, t2):
        """
        Add noise to samples between timesteps t1 and t2.
        
        Args:
            samples: Input samples [B, C, T, H, W]
            noise: Noise tensor [B, C, T, H, W]
            t1: Start timestep
            t2: End timestep
        """
        sigmas = extract_into_tensor(self.sigma_schedule, t1, samples.shape)
        alphas = extract_into_tensor(self.alpha_schedule, t1, samples.shape)
        sigmas_new = extract_into_tensor(self.sigma_schedule, t2, samples.shape)
        alphas_new = extract_into_tensor(self.alpha_schedule, t2, samples.shape)
        
        # Apply noise transformation
        samples = samples / alphas * alphas_new
        beta = sigmas_new ** 2 - (alphas_new / alphas * sigmas) ** 2
        beta = beta ** 0.5
        samples = samples + beta * noise
        
        return samples.to(samples.dtype)
    
    def obtain_mixed_noise(self, model_noise, noise, t1, t2):
        """
        Obtain mixed noise between model prediction and random noise.
        
        Args:
            model_noise: Noise predicted by the model [B, C, T, H, W]
            noise: Random noise tensor [B, C, T, H, W]
            t1: Start timestep
            t2: End timestep
        """
        sigmas = extract_into_tensor(self.sigma_schedule, t1, model_noise.shape)
        alphas = extract_into_tensor(self.alpha_schedule, t1, model_noise.shape)
        sigmas_new = extract_into_tensor(self.sigma_schedule, t2, model_noise.shape)
        alphas_new = extract_into_tensor(self.alpha_schedule, t2, model_noise.shape)
        
        # Calculate mixed noise
        beta = sigmas_new ** 2 - (alphas_new / alphas * sigmas) ** 2
        beta = beta ** 0.5
        mixed_noise = model_noise / alphas * alphas_new + beta * noise
        mixed_noise = mixed_noise / sigmas_new
        
        return mixed_noise
