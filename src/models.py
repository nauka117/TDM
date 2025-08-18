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


def generate_new(unet, noise_scheduler, latent, noise, encoder_hidden_states, prompt_attention_mask, 
                steps=1, return_mid=False, mid_points=None, total_steps=800, add_cfg=None):
    if add_cfg is not None:
        uncond_attention_mask = add_cfg['uncond_attention_mask']
        uncond_prompt_embeds = add_cfg['uncond_prompt_embeds']
        cfg = add_cfg['cfg']
    T_ = torch.randint(total_steps-1, total_steps, (latent.shape[0],), device=latent.device)
    T_ = T_.long()
    zero_t = torch.zeros_like(T_)
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod).to(latent.device).to(latent.dtype)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod).to(latent.device).to(latent.dtype)
    imgs_list = []
    # pure_noisy = noise_scheduler.add_noise(torch.randn_like(noise), noise, T_)
    pure_noisy = noise
    noisy_imgs_list = []
    added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
    for ind in range(steps):
        noisy_imgs_list.append(pure_noisy)
        noise_pred = unet(pure_noisy, timestep=T_, added_cond_kwargs=added_cond_kwargs,
                                            encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=prompt_attention_mask, return_dict=False)[0]
        noise_pred = noise_pred.chunk(2, dim=1)[0]
        if add_cfg is not None:
            noise_pred_uncond = unet(pure_noisy, timestep=T_, added_cond_kwargs=added_cond_kwargs,
                                                encoder_hidden_states=uncond_prompt_embeds, encoder_attention_mask=uncond_attention_mask, return_dict=False)[0]
            noise_pred_uncond = noise_pred_uncond.chunk(2, dim=1)[0] 
            noise_pred = noise_pred_uncond + cfg * (noise_pred - noise_pred_uncond)
        latent = predicted_origin(noise_pred,
                                    T_,
                                    pure_noisy,
                                    noise_scheduler.config.prediction_type,
                                    alpha_schedule,
                                    sigma_schedule,
                                )
        imgs_list.append(latent)
        if mid_points is not None:
            T_ = mid_points[ind+1] + zero_t
        else:
            T_ = T_ - total_steps // steps
        pure_noisy = noise_scheduler.add_noise(latent, noise_pred, T_).to(torch.float16)
    noisy_imgs_list.append(latent)
    if return_mid:
        return imgs_list, noisy_imgs_list
    return latent
