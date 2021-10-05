FROM ubuntu:18.04

WORKDIR /opt
COPY . /opt

USER root

RUN apt-get update
RUN apt-get install -y python3.8 \
                       python3-pip
# ADD requirements.txt .

RUN pip3 install -r ./src/requirements.txt

RUN apt-get update


ADD . .

ENTRYPOINT ["python3", "./src/train.py"]
