
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms

import numpy as np
from PIL import Image
from ActionEncoder import ActionEncoder

class PacmanBufferDataset(IterableDataset):
    def __init__(self, dataset, sequence_length=64):
        self.dataset = dataset
        self.sequence_length = sequence_length

        # Create a blank image with the same dimensions as your dataset images
        self.blank_image = Image.new('RGB', (552, 456), 'black')

        # Define a transform to convert PIL images to tensors
        self.transform = transforms.ToTensor()
        self.action_encoder = ActionEncoder()

    def __iter__(self):
        buffer = []

        for item in self.dataset:
            buffer.append(item)

            # If buffer is not full yet, pad with blank images
            if len(buffer) < self.sequence_length:
                current_sequence = {
                    'frame_images': torch.stack(
                        [self.transform(self.blank_image)] * (self.sequence_length - len(buffer)) +
                        [self.transform(b['frame_image']) for b in buffer]
                    ),
                    'actions': torch.tensor(
                        [self.action_encoder.encode(5)] * (self.sequence_length - len(buffer)) +
                        [self.action_encoder.encode(b['action']) for b in buffer]
                    ),
                    'episode': buffer[-1]['episode'],
                    'done': buffer[-1]['done']
                }
            else:
                # Once buffer is full, use sliding window
                current_sequence = {
                    'frame_images': torch.stack(
                        [self.transform(b['frame_image']) for b in buffer[-self.sequence_length:]]
                    ),
                    'actions': torch.tensor(
                        [self.action_encoder.encode(b['action']) for b in buffer[-self.sequence_length:]]
                    ),
                    'episode': buffer[-1]['episode'],
                    'done': buffer[-1]['done']
                }

            yield current_sequence

            # Optional: limit buffer size to save memory
            if len(buffer) >= self.sequence_length:
                buffer = buffer[-self.sequence_length:]
    
    def __len__(self):
        return self.dataset.info.splits['train'].num_examples
