FROM python:3.7-slim as base
ADD app/ /app

WORKDIR /app

RUN apt-get update
RUN apt-get install --assume-yes gcc
RUN apt-get -y install default-libmysqlclient-dev

RUN pip install --upgrade pip
RUN pip install torch==1.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir
RUN pip install pytorch-lightning --no-cache-dir
RUN pip install flask --no-cache-dir
RUN pip install flask-restx --no-cache-dir

RUN pip install mysqlclient --no-cache-dir

RUN pip install transformers --no-cache-dir
RUN pip install tensorflow-text --no-cache-dir
RUN pip install tensorflow-hub --no-cache-dir
RUN pip install py-eureka-client --no-cache-dir
RUN pip install Flask-SQLAlchemy --no-cache-dir
# RUN pip install -r requirements.txt

EXPOSE 5000
ENTRYPOINT [ "flask"]
CMD [ "run", "--host", "0.0.0.0" ]