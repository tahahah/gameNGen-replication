import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from peft import LoraConfig

class ModifiedUnet(nn.Module):

    def __init__(self, original_unet="segmind/tiny-sd", rank=64,  use_lora=True, use_dora=True, target_modules=None ):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(original_unet, subfolder="unet")

        self.unet.requires_grad_(False)


        if not None:
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"]


        if use_lora:
            self.lora_config = LoraConfig(
                r=rank,
                use_dora=use_dora,
                lora_alpha=64,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
        
            self.unet.add_adapter(self.lora_config)

    
    def forward(self, latent_model_input, time , encoder_hidden_states):
        x = self.unet(latent_model_input, time , encoder_hidden_states).sample
        return x
    # def check_lorelayer(self):
    #     lora_layers = filter(lambda p: p.requires_grad , self.model.parameters())

    