from torch.nn import nn
from diffusers import Unet2DConditionModel
from peft import LoraConfig

class ModifiedUnet(nn.Module):

    def __init__(self, original_unet="stabilityai/sdxl-turbo", rank=64, use_dora=True, target_modules=None):
        super().__init__()
        self.unet = Unet2DConditionModel(original_unet, subfolder="unet")
        self.unet.requires_grad_(False)

        if not target_modules:
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        
        self.lora_config = LoraConfig(
            r=rank,
            use_dora=use_dora,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )

    def initialize_lora(self):
        self.unet.add_adapter(self.lora_config)

    