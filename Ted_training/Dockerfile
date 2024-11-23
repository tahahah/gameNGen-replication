# Use the official CUDA 12.2 base image from NVIDIA
FROM nvidia/cuda:12.2.2-base-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Australia/Sydney

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    bzip2 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

RUN git clone https://github.com/eloialonso/diamond.git

WORKDIR /app/diamond

# Install the dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Install PyTorch with CUDA support
RUN pip3 install datasets python-dotenv huggingface-hub wandb torchvision

WORKDIR /app/diamond/src

# Copy the current directory contents into the container
COPY . /app/diamond/src

CMD ["python3", "train_model.py"]