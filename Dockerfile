FROM continuumio/miniconda3:22.11.1

# Install pip, pip-tools, and setuptools
RUN pip install --no-cache-dir --upgrade pip pip-tools setuptools

# Install PyTorch with CUDA support and openCV
RUN pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
RUN pip install opencv-python-headless
# Install pip packages from requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt-get --allow-releaseinfo-change update
RUN apt-get --allow-releaseinfo-change-suite update
RUN apt-get update
RUN mkdir /home/workdir
WORKDIR /home/workdir

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]
#CMD ["/bin/bash"]
