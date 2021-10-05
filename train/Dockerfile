
FROM python:3.8

WORKDIR /src

COPY . /src

# ADD requirements.txt .

RUN pip3 install -r /train/requirements.txt

RUN apt-get update && apt-get install -y libgl1

ADD . .

ENTRYPOINT ["python3", "./src/train.py"]
