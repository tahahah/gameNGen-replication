import json
import logging

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from peft import LoraConfig, get_peft_model


class ModifiedVAE(nn.Module):
    def __init__(self, original_vae="stabilityai/sdxl-turbo", target_modules=None, rank=64, use_dora=True, use_rslora=True, lora_dropout=0.05):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(original_vae, subfolder="vae")
        self.pre_encoder_layers = nn.Sequential()
        layers = [
            ('conv3d_initial', nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=0)),
            ('relu_initial', nn.ReLU()),
            ('pool3d_initial', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)),
            ('conv3d_middle', nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(3, 3, 3), stride=1, padding=0)),
            ('relu_middle', nn.ReLU()),
            ('pool3d_middle', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)),
            ('conv3d_final', nn.Conv3d(in_channels=16, out_channels=8, kernel_size=(3, 3, 3), stride=1, padding=0)),
            ('relu_final', nn.ReLU()),
            ('pool3d_final', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)),
            ('conv3d_output', nn.Conv3d(in_channels=8, out_channels=3, kernel_size=(1, 1, 1), stride=1, padding=0)),
            ('relu_output', nn.ReLU())
        ]
        for name, layer in layers:
            self.pre_encoder_layers.add_module(name, layer)

        if target_modules:
            self.target_modules = target_modules
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
        
        self.lora_config = LoraConfig(
            r=rank,
            use_dora=use_dora,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            use_rslora=use_rslora
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
