FROM python:3.8

WORKDIR /src

ADD requirements.txt .

RUN pwd

RUN pip3 install -r /src/requirements.txt

RUN apt-get update && apt-get install -y libgl1

ADD . .

ENTRYPOINT ["python3", "./src/train.py"]
