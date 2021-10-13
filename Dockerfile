FROM nvidia/cuda:11.1.1-base-ubuntu20.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
# RUN mkdir /app
ADD app/ /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh \
    && conda install -y python==3.8.3 \
    && conda clean -ya

# CUDA 11.1-specific steps
RUN conda install -y -c conda-forge cudatoolkit=11.1.1 \
    && conda install -y -c pytorch \
    "pytorch=1.8.1=py3.8_cuda11.1_cudnn8.0.5_0" \
    "torchvision=0.9.1=py38_cu111" \
    && conda clean -ya

RUN pip install torch
RUN pip install pytorch-lightning
RUN pip install transformers
RUN pip install -r requirements.txt

EXPOSE 5000
ENTRYPOINT [ "flask"]
CMD [ "run", "--host", "0.0.0.0" ]




# FROM pytorch/pytorch:latest

# ADD app/ /app
# WORKDIR /app

# RUN apt-get update \
#     && apt-get install -y \
#     libgl1-mesa-glx \
#     libx11-xcb1 \
#     && apt-get clean all \
#     && rm -r /var/lib/apt/lists/*

# RUN apt-get install mysql-client \
#     postgresql-client && \ 
#     default-libmysqlclient-dev

# RUN /opt/conda/bin/conda install --yes \
#     astropy \
#     matplotlib \
#     pandas \
#     scikit-learn \
#     scikit-image 

# RUN pip install torch
# RUN pip install pytorch-lightning
# RUN pip install transformers
# # RUN pip install flask
# # RUN pip install flask-rest
# RUN pip install -r requirements.txt

# EXPOSE 5000
# ENTRYPOINT [ "flask"]
# CMD [ "run", "--host", "0.0.0.0" ]

# FROM python:3
# ADD app/ /app
# WORKDIR /app
# # RUN pip3 install flask
# # RUN pip3 install flask-restx
# RUN pip3 install -r requirements.txt
# # RUN pip3 install torch
# RUN pip3 install https://download.pytorch.org/whl/cu90/torch-1.1.0-cp36-cp36m-win_amd64.whl
# RUN pip3 install pytorch-lightning --no-cache-dir
# RUN pip3 install transformers
# EXPOSE 5000
# ENTRYPOINT [ "flask"]
# CMD [ "run", "--host", "0.0.0.0" ]

# FROM pytorch/pytorch:latest

# ADD app/ /app
# WORKDIR /app

# RUN apt-get update \
#     && apt-get install -y \
#     libgl1-mesa-glx \
#     libx11-xcb1 \
#     && apt-get clean all \
#     && rm -r /var/lib/apt/lists/*

# RUN /opt/conda/bin/conda install --yes \
#     astropy \
#     matplotlib \
#     pandas \
#     scikit-learn \
#     scikit-image 

# RUN pip install torch
# RUN pip install pytorch-lightning
# RUN pip3 install transformers
# EXPOSE 5000
# ENTRYPOINT [ "flask"]
# CMD [ "run", "--host", "0.0.0.0" ]