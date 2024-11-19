import argparse
import gc
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from efficientvit.apps.utils.image import DMCrop
from efficientvit.ae_model_zoo import DCAE_HF
import os
import time

def process_image(dc_ae, image, resolution, device, output_path, model_name=""):
    """Process a single image with given resolution."""
    print(f"\nProcessing with model: mit-han-lab/{model_name}")
    print(f"Resolution: {resolution}px")
    
    transform = transforms.Compose([
        DMCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Save resized input image
    x = transform(image)
    filename, ext = os.path.splitext(output_path)
    input_filename = f"{filename}_{model_name}_{resolution}px_input{ext}"
    save_image(x, input_filename)
    print(f"Saved resized input image to: {input_filename}")
    
    # Add batch dimension and move to device
    x = x[None].to(device)
    
    # Time encoder
    start_time = time.time()
    latent = dc_ae.encode(x)
    encode_time = time.time() - start_time
    print(f"Encoding time: {encode_time:.3f} seconds")
    print(f"Latent shape: {latent.shape}")
    
    # Time decoder
    start_time = time.time()
    y = dc_ae.decode(latent)
    decode_time = time.time() - start_time
    print(f"Decoding time: {decode_time:.3f} seconds")
    
    # Save output
    filename, ext = os.path.splitext(output_path)
    resolution_output = f"{filename}_{model_name}_{resolution}px{ext}"
    save_image(y * 0.5 + 0.5, resolution_output)
    print(f"Saved output image to: {resolution_output}")
    print(f"Total inference time: {encode_time + decode_time:.3f} seconds")
    
    return x, latent, y

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run DC-AE model with custom parameters')
    parser.add_argument('--model_name', type=str, default="dc-ae-f32c32-in-1.0",
                      help='Name of the DC-AE model to use')
    parser.add_argument('--resolutions', type=int, nargs='+', default=[512],
                      help='List of resolutions for DMCrop (e.g., --resolutions 256 512 1024)')
    parser.add_argument('--input_image', type=str, default="pacman_og.png",
                      help='Input image path')
    parser.add_argument('--output_image', type=str, default="pacman.jpg",
                      help='Output image path')
    
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Loading model: mit-han-lab/{args.model_name}")
    
    # Load model
    start_time = time.time()
    dc_ae = DCAE_HF.from_pretrained("mit-han-lab/" + args.model_name)
    dc_ae = dc_ae.to(device).eval()
    
    # Count model parameters
    total_params = sum(p.numel() for p in dc_ae.parameters())
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Model loaded in {time.time() - start_time:.3f} seconds")

    # Load input image
    image = Image.open(args.input_image).convert('RGB')
    print(f"Processing input image: {args.input_image}")
    
    # Process each resolution
    for resolution in args.resolutions:
        try:
            x, latent, y = process_image(dc_ae, image, resolution, device, args.output_image, args.model_name)
            
            # Clean up individual iteration variables
            del x, latent, y
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing resolution {resolution}: {str(e)}")
            continue

    # Final cleanup
    gc.collect()
    torch.cuda.empty_cache()
    try:
        del dc_ae
    except:
        print('Variables already cleaned up')

if __name__ == "__main__":
    main()