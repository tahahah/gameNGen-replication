import json
import logging

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from peft import LoraConfig, get_peft_model
import copy

class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.squeeze(x, dim=1)

class ModifiedVAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pre_encoder_layers = nn.Sequential()
        layers = [
            ('conv3d_initial', nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=1)), # (b, 32, 3, w, h)
            ('relu_initial', nn.ReLU()),
            ('conv3d_middle', nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(3, 3, 3), stride=1, padding=1)),  # (b, 16, 3, w, h)
            ('relu_middle', nn.ReLU()),
            ('conv3d_final', nn.Conv3d(in_channels=16, out_channels=8, kernel_size=(3, 3, 3), stride=1, padding=1)),    # (b, 8, 3, w, h)
            ('relu_final', nn.ReLU()),
            ('conv3d_output', nn.Conv3d(in_channels=8, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=0)),    # (b, 1, 3, w, h)
            ('relu_output', nn.ReLU()),
            ('squeeze', Squeeze(dim=2)),
            ('conv_in', copy.deepcopy(self.encoder.conv_in))
        ]
        for name, layer in layers:
            pre_encoder_layers.add_module(name, layer)
        self.encoder.conv_in = pre_encoder_layers

        if 'target_modules' in kwargs:
            self.target_modules = kwargs['target_modules']
        else:
            self.target_modules = [
                "conv_in",
                "down_blocks.0.resnets.0.conv1",
                "down_blocks.0.resnets.0.conv2",
                "down_blocks.0.resnets.1.conv1",
                "down_blocks.0.resnets.1.conv2",
                "down_blocks.0.downsamplers.0.conv",
                "down_blocks.1.resnets.0.conv1",
                "down_blocks.1.resnets.0.conv2",
                "down_blocks.1.resnets.0.conv_shortcut",
                "down_blocks.1.resnets.1.conv1",
                "down_blocks.1.resnets.1.conv2",
                "down_blocks.1.downsamplers.0.conv",
                "down_blocks.2.resnets.0.conv1",
                "down_blocks.2.resnets.0.conv2",
                "down_blocks.2.resnets.0.conv_shortcut",
                "down_blocks.2.resnets.1.conv1",
                "down_blocks.2.resnets.1.conv2",
                "down_blocks.2.downsamplers.0.conv",
                "down_blocks.3.resnets.0.conv1",
                "down_blocks.3.resnets.0.conv2",
                "down_blocks.3.resnets.1.conv1",
                "down_blocks.3.resnets.1.conv2",
                "mid_block.attentions.0.to_q",
                "mid_block.attentions.0.to_k",
                "mid_block.attentions.0.to_v",
                "mid_block.attentions.0.to_out.0",
                "mid_block.resnets.0.conv1",
                "mid_block.resnets.0.conv2",
                "mid_block.resnets.1.conv1",
                "mid_block.resnets.1.conv2",
                "conv_out"
            ]
        rank = kwargs["rank"] if "rank" in kwargs else 64
        self.lora_config = LoraConfig(
            r=rank,
            use_dora=(kwargs["use_dora"] if "use_dora" in kwargs else False),
            lora_alpha=rank,
            init_lora_weights="gaussian",
            lora_dropout=(kwargs["lora_dropout"] if "lora_dropout" in kwargs else None),
            target_modules=self.target_modules,
            use_rslora=(kwargs["use_rslora"] if "use_rslora" in kwargs else None)
        )
        # self.initialize_encoder_lora()
        # self.freeze_decoder()

    def initialize_encoder_lora(self):
        get_peft_model(self.vae.encoder, self.lora_config)

    def freeze_decoder(self):
        for name, param in self.named_parameters():
            if "decoder" in name:
                param.requires_grad = False
    
    def log_trainable_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}, Total Trainable parameters: {total_trainable_params}")


    def encode(self, x):
        x = self.pre_encoder_layers(x)

        return self.vae.encode(x)
    
    def decode(self, latents):
        # Use the original VAE decoder
        return self.vae.decode(latents)
