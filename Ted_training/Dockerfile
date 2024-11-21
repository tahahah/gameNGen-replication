# Use the official CUDA 12.2 base image from NVIDIA
FROM nvidia/cuda:12.2.2-base-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Australia/Sydney

# Install Miniconda and required packages
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh \
    && bash /miniconda.sh -b -p /opt/conda \
    && rm /miniconda.sh \
    && /opt/conda/bin/conda clean -a

# Update PATH environment variable
ENV PATH=/opt/conda/bin:$PATH

# Install Python 3.8 and PyTorch nightly build with CUDA 12.2
RUN conda install python=3.8.2 -y \
    && conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch-nightly -c nvidia -y

#
RUN cd /opt/conda/lib \
    && mkdir backup \
    && mv libstd* backup \
    && cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ./ \
    && ln -s libstdc++.so.6 libstdc++.so \
    && ln -s libstdc++.so.6 libstdc++.so.6.0.19

# Copy the uv tool from the GitHub container registry
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV UV_SYSTEM_PYTHON=1

# Set the working directory in the container
WORKDIR /app

RUN git clone https://github.com/eloialonso/diamond.git

WORKDIR /app/diamond

# Install the dependencies from requirements.txt
RUN pip install -r requirements.txt
RUN pip install datasets python-dotenv huggingface-hub wandb

WORKDIR /app/diamond/src

# Copy the current directory contents into the container
COPY . /app/diamond/src

CMD ["python", "train_model.py"]