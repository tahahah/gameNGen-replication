import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from peft import LoraConfig


class ModifiedUNet(UNet2DConditionModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # unet = super().from_pretrained("segmind/tiny-sd", subfolder="unet")
        self.register_to_config(cross_attention_dim=256)

        for down_block in self.down_blocks:
            for attn in down_block.attentions:
                for transformer_block in attn.transformer_blocks:
                    if hasattr(transformer_block, "attn2"):
                        transformer_block.attn2.to_k = torch.nn.Linear(256, transformer_block.attn2.to_k.out_features, bias=False)
                        transformer_block.attn2.to_v = torch.nn.Linear(256, transformer_block.attn2.to_v.out_features, bias=False)

        for up_block in self.up_blocks:
            for attn in up_block.attentions:
                for transformer_block in attn.transformer_blocks:
                    if hasattr(transformer_block, "attn2"):
                        transformer_block.attn2.to_k = torch.nn.Linear(256, transformer_block.attn2.to_k.out_features, bias=False)
                        transformer_block.attn2.to_v = torch.nn.Linear(256, transformer_block.attn2.to_v.out_features, bias=False)