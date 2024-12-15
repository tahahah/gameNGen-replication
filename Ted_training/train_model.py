"""
Training script for the Denoiser model using PacmanBuffer dataset.
This module handles the training loop and model configuration.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from datasets import load_dataset
from pathlib import Path
from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
import wandb
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import login, upload_file, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from models.diffusion import Denoiser
from models.diffusion import DiffusionSampler
from utils import extract_state_dict
from PacmanBufferDataset import PacmanBufferDataset


@dataclass
class InnerModelConfig:
    """Configuration for the inner model architecture."""
    img_channels: int = 3
    num_steps_conditioning: int = 4
    cond_channels: int = 100
    depths: list[int] = (2, 2, 2, 2)
    channels: list[int] = (16, 16, 16, 16)
    attn_depths: list[int] = (0, 0, 0, 0)
    num_actions: int = 4


@dataclass
class DenoiserConfig:
    """Configuration for the Denoiser model."""
    sigma_data: float = 0.5
    sigma_offset_noise: float = 0.3
    inner_model: InnerModelConfig = InnerModelConfig()


@dataclass
class SigmaConfig:
    """Configuration for sigma distribution parameters."""
    loc: float = -0.4
    scale: float = 1.2
    sigma_min: float = 2e-3
    sigma_max: float = 20


@dataclass
class Batch:
    """Data batch container for training."""
    obs: torch.ByteTensor
    act: torch.LongTensor
    mask_padding: torch.BoolTensor

    def pin_memory(self) -> Batch:
        """Pin batch tensors to memory for faster GPU transfer."""
        return Batch(**{
            k: v if k in ("segment_ids", "info") else v.pin_memory() 
            for k, v in self.__dict__.items()
        })

    def to(self, device: torch.device) -> Batch:
        """Move batch tensors to specified device."""
        return Batch(**{
            k: v if k in ("segment_ids", "info") else v.to(device) 
            for k, v in self.__dict__.items()
        })


def create_batch_from_obs(obs: Dict[str, torch.Tensor], device: torch.device) -> Batch:
    """
    Create a Batch instance from observation dictionary.
    
    Args:
        obs: Dictionary containing episode data
        device: Target device for tensors
    
    Returns:
        Batch: Processed batch ready for training
    """
    frames = obs['frame_images']
    actions = obs['actions']
    
    return Batch(
        obs=frames.type(torch.float16),
        act=actions.type(torch.LongTensor),
        mask_padding=torch.full((1, 16), True, dtype=torch.bool)
    ).to(device)


def log_images(batch: Batch, pred_images: torch.Tensor, step: int, max_images: int = 8) -> None:
    """
    Log a batch of images to WandB.
    
    Args:
        batch: Current training batch
        pred_images: Predicted images from the model
        step: Current training step
        max_images: Maximum number of images to log
    """
    # Get the real images from the batch
    real_images = batch.obs[:max_images]
    pred_images = pred_images[:max_images]
    
    # Convert to grid
    real_grid = make_grid(real_images, nrow=4, normalize=True, value_range=(0, 255))
    pred_grid = make_grid(pred_images, nrow=4, normalize=True)
    
    # Convert to numpy for wandb
    real_grid = real_grid.cpu().numpy().transpose(1, 2, 0)
    pred_grid = pred_grid.cpu().numpy().transpose(1, 2, 0)
    
    # Log to wandb
    wandb.log({
        "images/real": wandb.Image(real_grid, caption="Real Images"),
        "images/predicted": wandb.Image(pred_grid, caption="Predicted Images"),
    }, step=step)


def save_to_huggingface(checkpoint_path: str, model_name: str) -> None:
    """
    Save model checkpoint to HuggingFace Hub.
    
    Args:
        checkpoint_path: Path to the local checkpoint file
        model_name: Name of the model on HuggingFace Hub
    """
    # Login to HuggingFace
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        logging.warning("HF_TOKEN not found in environment variables. Skipping HuggingFace upload.")
        return
        
    login(token=hf_token)
    
    # Prepare repository details
    repo_id = f"Tahahah/{model_name}"
    path_in_repo = f"checkpoints/{os.path.basename(checkpoint_path)}"
    
    try:
        # Try to upload directly
        upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model"
        )
    except RepositoryNotFoundError:
        # If repository doesn't exist, create it and then upload
        create_repo(repo_id, repo_type="model")
        upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model"
        )
    
    logging.info(f"Model checkpoint uploaded to HuggingFace: {repo_id}/{path_in_repo}")


def train_diffusion_model(
    denoiser: Denoiser,
    train_loader: DataLoader,
    optimizer: torch.optim.AdamW,
    num_epochs: int,
    device: torch.device,
    log_interval: int = 1,
    image_log_interval: int = 100,
    model_name: str = "pacman-ted-denoiser"
) -> None:
    """
    Training loop for diffusion model using Denoiser.
    
    Args:
        denoiser: The denoiser model to train
        train_loader: DataLoader providing training batches
        optimizer: Optimizer for updating model parameters
        num_epochs: Number of training epochs
        device: Device to run training on
        log_interval: Number of steps between logging metrics
        image_log_interval: Number of steps between logging images
        model_name: Name of the model for HuggingFace Hub
    """
    denoiser.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            global_step = epoch * len(train_loader) + batch_idx
            
            # Process batch
            batch_train = create_batch_from_obs(batch, device)
            
            # Training step
            optimizer.zero_grad()
            loss, logs = denoiser(batch_train)
            
            # Get predicted images if it's time to log images
            if global_step % image_log_interval == 0:
                with torch.no_grad():
                    # Use a small sigma for inference to get cleaner predictions
                    batch_size = batch_train.obs.size(0)
                    sigma = torch.ones(batch_size, device=denoiser.device) * 0.1  # Small sigma for clean predictions
                    n = denoiser.cfg.inner_model.num_steps_conditioning
                    
                    # Get the conditioning observations and actions
                    obs = batch_train.obs[:, :n]  # Take first n frames for conditioning
                    b, t, c, h, w = obs.shape
                    obs = obs.reshape(b, t * c, h, w)
                    act = batch_train.act[:, :n]
                    
                    # Create a new batch for logging with just the target frame
                    log_batch = Batch(
                        obs=batch_train.obs[:, n],
                        act=batch_train.act[:, n:n+1],
                        mask_padding=batch_train.mask_padding[:, n:n+1]
                    )
                    
                    # Get the next observation to denoise
                    noisy_next_obs = denoiser.apply_noise(log_batch.obs, sigma, denoiser.cfg.sigma_offset_noise)
                    
                    # Denoise the observation
                    pred_images = denoiser.denoise(noisy_next_obs, sigma, obs, act)
                    log_images(log_batch, pred_images, global_step)

                    # Log the full sequence from first batch every 16 steps
            if global_step % 16 == 0:
                seq_length = batch_train.obs.size(1)
                sequence_batch = Batch(
                    obs=batch_train.obs[0, :seq_length],  # All frames from first sequence
                    act=batch_train.act[0, :seq_length],
                    mask_padding=batch_train.mask_padding[0, :seq_length]
                )
                # Log sequence without predictions
                wandb.log({
                    "sequence/frames": wandb.Image(
                        make_grid(sequence_batch.obs, nrow=seq_length, normalize=True, value_range=(0, 255)).cpu(),
                        caption=f"Full Sequence at step {global_step}"
                    )
                }, step=global_step)
            
            loss.backward()
            optimizer.step()
            
            # Logging metrics
            epoch_loss += loss.item()
            if batch_idx % log_interval == 0:
                metrics = {
                    "train/loss": loss.item(),
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                }
                if "loss_denoising" in logs:
                    metrics["train/loss_denoising"] = logs["loss_denoising"]
                
                wandb.log(metrics, step=global_step)
                logging.info(f"Step {global_step}: {metrics}")
        
        # Epoch summary
        avg_loss = epoch_loss / len(train_loader)
        epoch_metrics = {
            "train/epoch_avg_loss": avg_loss,
            "train/epoch": epoch,
        }
        wandb.log(epoch_metrics)
        logging.info(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint and generate samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Create checkpoint directory
            os.makedirs("checkpoints", exist_ok=True)
            
            # Save checkpoint locally
            checkpoint_path = f"checkpoints/model_epoch_{epoch}.pt"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': denoiser.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': {
                    'model_name': model_name,
                    'epoch': epoch,
                    'global_step': global_step,
                }
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Upload to HuggingFace
            save_to_huggingface(checkpoint_path, model_name)
            
            # Save to WandB
            wandb.save(checkpoint_path)
            
            # Generate and log samples
            denoiser.eval()
            with torch.no_grad():
                sample_batch = next(iter(train_loader))
                sample_batch = create_batch_from_obs(sample_batch, device)
                samples = denoiser.sample(sample_batch)
                log_images(sample_batch, samples, global_step, max_images=16)
            denoiser.train()


def main():
    """Main training script entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Setup environment variables
    load_dotenv()
    
    # Setup wandb
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Initialize model and configurations
    cfg = DenoiserConfig()
    denoiser = Denoiser(cfg)
    denoiser.to(device)
    
    # Setup training parameters
    sigma_config = SigmaConfig()
    denoiser.setup_training(sigma_config)
    
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=1e-4)
    
    # Model name for HuggingFace
    model_name = "pacman-ted-denoiser"

    # Dataset
    dataset = load_dataset("Tahahah/PacmanDataset_3", split="train", verification_mode="no_checks", streaming=True)
    
    # Create the dataset
    buffer_dataset = PacmanBufferDataset(dataset, sequence_length=16)

    # Create the DataLoader
    dataloader = DataLoader(
        buffer_dataset,
        batch_size=1,
        num_workers=0,  # Reduce worker processes to avoid shared memory issues
        pin_memory=True,
        persistent_workers=False  # Disable persistent workers
    )
    
    # Initialize wandb with config
    wandb.init(
        project="PacmanTedTrain",
        magic=True,
        config={
            "model_config": asdict(cfg),
            "sigma_config": asdict(sigma_config),
            "optimizer": optimizer.__class__.__name__,
            "learning_rate": 1e-4,
            "device": str(device),
            "num_epochs": 70000,
            "log_interval": 1,
            "image_log_interval": 100,
            "model_name": model_name,
        }
    )
    
    # Start training
    try:
        train_diffusion_model(
            denoiser=denoiser,
            train_loader=dataloader,
            optimizer=optimizer,
            num_epochs=70000,
            device=device,
            model_name=model_name
        )
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()