import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
import numpy as np

def parse_filename(filename):
    # Extract model name and resolution from filename
    base = os.path.basename(filename)
    parts = base.replace('output_', '').replace('.jpg', '').split('_')
    model_name = '_'.join(parts[:-1])  # Everything except the last part
    resolution = int(parts[-1].replace('px', ''))  # Last part contains resolution
    return model_name, resolution

def plot_comparison():
    # Get all output images (from current directory)
    image_files = glob.glob('output_*.jpg')
    print(f"Found files: {image_files}")
    
    if not image_files:
        print("No images found! Make sure the output images are in the current directory.")
        return
    
    # Extract unique model names and resolutions
    model_names = sorted(list(set(parse_filename(f)[0] for f in image_files)))
    resolutions = sorted(list(set(parse_filename(f)[1] for f in image_files)))
    
    print(f"Models: {model_names}")
    print(f"Resolutions: {resolutions}")
    
    # Create figure
    n_models = len(model_names)
    n_resolutions = len(resolutions)
    
    # Calculate figure size (adjust these multipliers as needed)
    fig, axes = plt.subplots(n_models, n_resolutions, figsize=(n_resolutions * 4, n_models * 4))
    
    # Create a mapping of model names to shorter labels if names are too long
    model_labels = {name: f"Model {i+1}\n{name}" for i, name in enumerate(model_names)}
    
    # Plot images in a grid
    for i, model in enumerate(model_names):
        for j, res in enumerate(resolutions):
            current_file = f'output_{model}_{res}px.jpg'
            print(f"Looking for file: {current_file}")
            
            if os.path.exists(current_file):
                print(f"Found file: {current_file}")
                # Get the correct subplot
                if n_models == 1:
                    ax = axes[j] if n_resolutions > 1 else axes
                elif n_resolutions == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]
                
                # Read and plot image
                img = Image.open(current_file)
                ax.imshow(np.array(img))
                
                # Only show title for top row
                if i == 0:
                    ax.set_title(f'{res}px')
                
                # Only show model name for leftmost column
                if j == 0:
                    ax.set_ylabel(model_labels[model], rotation=0, labelpad=50, ha='right')
                
                # Remove ticks
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                print(f"File not found: {current_file}")
    
    # Adjust layout to prevent overlap
    plt.tight_layout(h_pad=2, w_pad=2)
    
    # Save the plot
    plt.savefig('comparison_matrix.png', dpi=300, bbox_inches='tight')
    print("Plot saved as comparison_matrix.png")

plot_comparison()