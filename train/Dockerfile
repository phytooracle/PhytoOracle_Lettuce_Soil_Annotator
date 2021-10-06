FROM python:3.8

RUN pip install --upgrade pip

WORKDIR /src
COPY . /src

COPY ./requirements.txt .

RUN ls

RUN pip3 install -r requirements.txt

RUN apt-get update && apt-get install -y libgl1

COPY . .

ENTRYPOINT ["python3", "./src/train.py"]
