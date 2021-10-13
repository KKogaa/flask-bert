#!/bin/bash
sudo apt update && sudo apt upgrade -Y
apt-get install -y libmysqlclient-dev
# pip install torch
# pip install pytorch-lightning
# pip install transformers
cd app
pip install gdown
gdown https://drive.google.com/uc?id=1-MoqQMRdVWLb3GqNDxIvAwQTjlM2kGas
pip install -r requirements.txt

