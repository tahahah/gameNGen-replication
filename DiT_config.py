@dataclass
class DiTConfig:
    name: str = "DiT"

    input_size: int = 16
    patch_size: int = 2
    in_channels: int = 32
    hidden_size: int = 100
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    post_norm: bool = False
    class_dropout_prob: float = 0.1
    num_classes: int = 5
    learn_sigma: bool = True
    unconditional: bool = False
    use_checkpoint: bool = True

    pretrained_path: Optional[str] = None
    pretrained_source: str = "dc-ae"  # dit

    eval_scheduler: str = "GaussianDiffusion"
    num_inference_steps: int = 250
    train_scheduler: str = "GaussianDiffusion"