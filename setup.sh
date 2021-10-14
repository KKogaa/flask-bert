#!/bin/bash
sudo apt update && sudo apt upgrade -Y
sudo apt-get install -y python3 
sudo apt-get install -y python3-pip
sudo apt-get install -y libmysqlclient-dev
# pip install torch
# pip install pytorch-lightning
# pip install transformers
cd app
pip install gdown
gdown https://drive.google.com/uc?id=1-MoqQMRdVWLb3GqNDxIvAwQTjlM2kGas
# pip install -r requirements.txt
pip3 install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip3 install pytorch-lightning
pip3 install transformers

