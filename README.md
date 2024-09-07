Modify SDXL Turbo's architecture to accept frame history and action embeddings.
Implement the frame history encoder and action embedding layers.
Apply DoRA to the appropriate layers of the modified model.
Collect a large dataset of Pacman gameplay.
Implement the training loop, including the noise addition process described in the GameNGen paper.
Fine-tune the model using DoRA where applicable.

SDXL Turbo
- VAE Encoder weights: https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/vae_encoder/model.onnx
- VAE Decoder weights: https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/vae_encoder/model.onnx
