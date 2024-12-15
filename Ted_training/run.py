# from models.diffusion import Denoiser, DiffusionSampler
import pygame
from dataclasses import dataclass, asdict
from ActionEncoder import ActionEncoder
import torch
from models.diffusion import Denoiser
from models.diffusion import DiffusionSampler
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms



@dataclass
class DiffusionSamplerConfig:
    num_steps_denoising: int = 3
    sigma_min: float = 2e-3
    sigma_max: float = 5
    rho: int = 7
    order: int = 1
    s_churn: float = 0
    s_tmin: float = 0
    s_tmax: float = float("inf")
    s_noise: float = 1

class InnerModelConfig:
    img_channels: int = 3
    num_steps_conditioning: int = 4
    cond_channels: int = 100
    depths: list = [2, 2, 2, 2]
    channels: list = [16, 16, 16, 16]
    attn_depths: list = [0, 0, 0, 0]
    num_actions: int = 4


class DenoiserConfig:
  sigma_data = 0.5
  sigma_offset_noise =  0.3
  inner_model = InnerModelConfig()

def game_init():
    # Load the PNG image
    # Load the PNG image
    
    # Specify the paths to your 4 PNG files
    image_paths = ["output_image0.png", "output_image1.png", "output_image2.png", "output_image3.png"]

    # Define a transform to resize and convert images to tensors
    transform = transforms.Compose([
        transforms.Resize((456, 552)), 
        transforms.ToTensor(),          # Convert image to tensor (C, H, W)
    ])

    # Load and transform each image
    image_tensors = []
    for path in image_paths:
        img = Image.open("init_db/" +path).convert("RGB")  # Ensure the image has 3 channels (RGB)
        img_tensor = transform(img)           # Shape: (3, 456, 552)
        image_tensors.append(img_tensor)

    # Stack the tensors to create a batch with shape (4, 3, 456, 552)
    batch_tensor = torch.stack(image_tensors)  # Shape: (4, 3, 456, 552)

    # Add an extra batch dimension to create shape (1, 4, 3, 456, 552)
    final_tensor = batch_tensor.unsqueeze(0)  # Shape: (1, 4, 3, 456, 552)

    print(final_tensor.shape)
    return final_tensor
#     cfg = DenoiserConfig()
#     denoiser = Denoiser(cfg)
#     diffSampleConfig = DiffusionSamplerConfig()
#     sampler = DiffusionSampler(denoiser, diffSampleConfig)
#     return sampler



def game_run(diffsamp : DiffusionSampler , Images : torch.Tensor):
    # Initialize Pygame
    pygame.init()


    # Action Stuff
    AE = ActionEncoder()
    action_tensor = torch.tensor([[0 , 0, 0, 0]])
    action_t = 'left'
    
    # Set up a screen
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Key Press Detection")

    # Game loop
    running = True
    while running:
        for event in pygame.event.get():
            # Check if the user wants to quit
            if event.type == pygame.QUIT:
                running = False

        # Check if specific keys are pressed
        keys = pygame.key.get_pressed()  # Get the current state of all keys
        if event.type == pygame.KEYDOWN:  # Check if a key is pressed
            if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                if event.key == pygame.K_UP:
                    print("Up arrow key pressed!")
                    action_t = "up"
                elif event.key == pygame.K_DOWN:
                    print("Down arrow key pressed!")
                    action_t = "down"
                elif event.key == pygame.K_LEFT:
                    print("Left arrow key pressed!")
                    action_t = "left"
                elif event.key == pygame.K_RIGHT:
                    print("Right arrow key pressed!")
                    action_t = "right"
        print(AE.encode(action_t))
        action_tensor = torch.cat((action_tensor[0][1:], torch.tensor([AE.encode(action_t)]))).unsqueeze(dim=0)
        print(action_tensor)
        
        x, traj = diffsamp.sample(Images,action_tensor)

        # Step 1: Remove the first image ([0][0])
        Shifted_Images = Images[:, 1:, :, :, :]  # Shape: (1, 3, 3, 456, 552)

        # Step 2: Add the new image at the end
        new_image = x.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions, shape: (1, 1, 3, 456, 552)
        Images = torch.cat((Shifted_Images, new_image), dim=1)  # Shape: (1, 4, 3, 456, 552)

        plt.imshow(new_image)
        plt.show()

        if keys[pygame.K_ESCAPE]:  # ESC to quit
            running = False
        

    # Quit Pygame
    pygame.quit()



def main():
    images = game_init()

    denoiser_cfg = DenoiserConfig()
    denoiser = Denoiser(denoiser_cfg)

    diffusion_sampler_cfg = DiffusionSamplerConfig()
    diffsamp = DiffusionSampler(denoiser , diffusion_sampler_cfg)

    game_run(diffsamp , images)



    # for i in range(4):
    #     diimage = images[0][i].permute(1, 2, 0)  # Rearrange to (456, 552, 3) for display
    #     plt.imshow(diimage)  # Display the image
    #     plt.axis('off')  # Turn off axes
    #     plt.pause(1)
    

if __name__ == "__main__":
    main()