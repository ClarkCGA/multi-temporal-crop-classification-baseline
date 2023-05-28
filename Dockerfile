ARG PYTORCH="2.0.0"
ARG TORCHVISION = "0.15.0"
ARG CUDA="11.7"

FROM continuumio/miniconda3:4.8.3

# Install pip, pip-tools, and setuptools
RUN pip install --no-cache-dir --upgrade pip pip-tools setuptools

# Install basic dependencies and nvidia-container-toolkit
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    wget \
    nvidia-container-toolkit \
    && rm -rf /var/lib/apt/lists/*

RUN nvidia-container-runtime configure --disable-cgroups
RUN systemctl restart docker

# Install PyTorch with CUDA support
RUN conda install -y pytorch=${PYTORCH} torchvision=${TORCHVISION} pytorch-cuda=${CUDA} -c pytorch -c nvidia

# Install pip packages from requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir /home/workdir
WORKDIR /home/workdir

EXPOSE 8888

CMD ["/bin/bash"]