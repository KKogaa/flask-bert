#!/bin/bash
sudo apt update && sudo apt upgrade -Y
pip install torch
pip install pytorch-lightning
pip install transformers
cd app
pip install gdown
gdown https://drive.google.com/drive/folders/1-I0bwTOaPT1Ei4WLYNioTlYHweio04pb
pip install -r requirements.txt

