FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Install system packages
RUN apt-get update --fix-missing && \
  apt-get install -y \
  wget \
  libgtk2.0-dev \
  bzip2 \
  ca-certificates \
  curl \
  git \
  vim \
  g++ \
  gcc \
  graphviz \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libglib2.0-0 \
  libgl1-mesa-glx \
  libhdf5-dev \
  openmpi-bin \
  && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

COPY ./experiments_food/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY experiments_food ./
