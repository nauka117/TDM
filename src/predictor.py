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
                return_double=False, timestep_cond=None):
        for _ in range(steps):
            score_pred = score_model(noisy_samples, timestep=timesteps, added_cond_kwargs=self.added_cond_kwargs,
                                                    encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=prompt_attention_mask, return_dict=False)[0]
            score_pred = score_pred.chunk(2, dim=1)[0]
            # score_pred = score_model(noisy_samples, timesteps, encoder_hidden_states, timestep_cond = timestep_cond, return_dict=False)[0]
            if cfg is not None:
                score_uncon_pred = score_model(noisy_samples, timestep=timesteps, added_cond_kwargs=self.added_cond_kwargs,
                                encoder_hidden_states=self.uncond_prompt_embeds, encoder_attention_mask=self.uncond_attention_mask, return_dict=False)[0]
                score_uncon_pred = score_uncon_pred.chunk(2, dim=1)[0]
                # score_pred_cfg = score_pred + cfg * (score_pred - score_uncon_pred)
                score_pred_cfg = score_uncon_pred + cfg * (score_pred - score_uncon_pred)
                pred_latents = predicted_origin(
                    score_pred_cfg,
                    timesteps.long(),
                    noisy_samples,
                    self.noise_scheduler.config.prediction_type,
                    self.alpha_schedule,
                    self.sigma_schedule, )
                pred_latents_nocfg = predicted_origin(
                    score_pred,
                    timesteps.long(),
                    noisy_samples,
                    self.noise_scheduler.config.prediction_type,
                    self.alpha_schedule,
                    self.sigma_schedule, )
                timesteps = timesteps - timesteps // steps
                noisy_samples = self.noise_scheduler.add_noise(pred_latents.detach(), score_pred_cfg, timesteps.long())
                if return_double:
                    return score_pred_cfg, pred_latents, pred_latents_nocfg
            else:
                pred_latents = predicted_origin(
                    score_pred,
                    timesteps,
                    noisy_samples,
                    self.noise_scheduler.config.prediction_type,
                    self.alpha_schedule,
                    self.sigma_schedule, )
                timesteps = timesteps - timesteps // steps
                noisy_samples = self.noise_scheduler.add_noise(pred_latents.detach(), score_pred, timesteps)
        if cfg is not None:
            return score_pred_cfg, pred_latents
        else:
            return score_pred, pred_latents
        
    def add_noise(self, samples, noise, t1, t2):
        sigmas = extract_into_tensor(self.sigma_schedule, t1, samples.shape)
        alphas = extract_into_tensor(self.alpha_schedule, t1, samples.shape)
        sigmas_new = extract_into_tensor(self.sigma_schedule, t2, samples.shape)
        alphas_new = extract_into_tensor(self.alpha_schedule, t2, samples.shape)
        samples = samples / alphas * alphas_new # x = alphas_new * x + alphas_new / alphas * sigmas * eps
        beta = sigmas_new ** 2 - ( alphas_new / alphas * sigmas ) ** 2
        beta = beta ** 0.5
        samples = samples + beta * noise
        return samples.to(torch.float16)
    
    def obtain_mixed_noise(self, model_noise, noise, t1, t2):
        sigmas = extract_into_tensor(self.sigma_schedule, t1, model_noise.shape)
        alphas = extract_into_tensor(self.alpha_schedule, t1, model_noise.shape)
        sigmas_new = extract_into_tensor(self.sigma_schedule, t2, model_noise.shape)
        alphas_new = extract_into_tensor(self.alpha_schedule, t2, model_noise.shape)
        
        beta = sigmas_new ** 2 - ( alphas_new / alphas * sigmas ) ** 2
        beta = beta ** 0.5
        mixed_noise = model_noise / alphas * alphas_new + beta * noise
        mixed_noise = mixed_noise / sigmas_new
        return mixed_noise
