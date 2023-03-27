from share import *
import config

import numpy as np
import torch
import random
import base64
import json
import sys
import io, os

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

model_dir = "/opt/ml/model"

def input_fn(request_body, request_content_type):
    
    assert request_content_type=='application/json'
    data = json.loads(request_body)
    return data
        
def model_fn(model_dir):
        
    model = create_model(f'{model_dir}/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(f'{model_dir}/control_sd15_hed.pth', location='cuda'))
    model = model.cuda()
    return model
    

def predict_fn(input_object, model):

    body = input_object
    H = body['H'] 
    W = body['W'] 
    C = body['C']

    detected_map = np.reshape(
        np.frombuffer(
            base64.decodebytes(bytes(body['detected_map'], encoding="utf-8")),
            dtype=np.uint8,
        ),
        (H, W, C),
    )

    prompt = body['prompt'] 
    a_prompt = body['a_prompt']
    n_prompt = body['n_prompt'] 
    num_samples = body['num_samples']  
    ddim_steps = body['ddim_steps']  
    guess_mode = body['guess_mode']  
    strength = body['strength']  
    scale = body['scale']  
    seed = body['seed']  
    eta = body['eta']  

    dim_sampler = DDIMSampler(model)
    with torch.no_grad():
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = {"detected_map": detected_map.tolist(),
            "image": x_samples.tolist()
            }
    return results
    
def output_fn(predictions, content_type):
    assert content_type == 'application/json'
    return json.dumps(predictions)


