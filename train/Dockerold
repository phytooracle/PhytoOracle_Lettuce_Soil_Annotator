FROM python:3.8

RUN pip install --upgrade pip

WORKDIR /opt
COPY . /opt


RUN ls

COPY requirements.txt .

RUN pip3 install -r requirements.txt

RUN apt-get update && apt-get install -y libgl1

COPY . .

ENTRYPOINT ["python3", "./opt/train.py"]
