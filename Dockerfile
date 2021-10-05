FROM python:3
ADD app/ /app
WORKDIR /app
RUN pip3 install flask
RUN pip3 install flask-restx
RUN pip3 install torch --no-cache-dir
RUN pip3 install pytorch-lightning --no-cache-dir
RUN pip3 install transformers
# RUN pip3 install -r requirements.txt
EXPOSE 5000
ENTRYPOINT [ "flask"]
CMD [ "run", "--host", "0.0.0.0" ]

# docker build --tag flask-docker-demo-app .
# docker run -p 5000:5000 flask-docker-demo-app

# docker run -p 5000:5000 flask-docker-demo-poop
# FROM ubuntu:20.04
# RUN apt-get update -ye && \
# 	apt-get install -y python-pip python-dev
# RUN pip install --upgrade pip
# ADD app/ /app
# WORKDIR /app
# RUN pip3 install -r requirements.txt
# EXPOSE 5000
# CMD [ "python3 app.py" ]

# FROM pytorch/pytorch:latest

# ADD app/ /app
# WORKDIR /app

# RUN apt-get update \
# 	&& apt-get install -y \
# 	libgl1-mesa-glx \
# 	libx11-xcb1 \
# 	&& apt-get clean all \
# 	&& rm -r /var/lib/apt/lists/*

# RUN /opt/conda/bin/conda install --yes \
# 	astropy \
# 	matplotlib \
# 	pandas \
# 	scikit-learn \
# 	scikit-image 

# RUN pip install flask
# RUN pip install flask-restx
# RUN pip install torch
# RUN pip install pytorch-lightning
# RUN pip install transformers

# EXPOSE 5000
# CMD [ "python app.py" ]