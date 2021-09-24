FROM python:3.8

WORKDIR /src

ADD requirements.txt .

RUN pip3 install -r requirements.txt
RUN pip3 install torch-points-kernels
RUN pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
RUN pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
RUN pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
RUN pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
RUN pip3 install torch-geometric

RUN apt-get update && apt-get install -y libgl1

ADD . .

ENTRYPOINT ["python3", "./main.py"]