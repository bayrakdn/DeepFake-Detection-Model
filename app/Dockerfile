#FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 not sup GTX 1650
#FROM python:3.10-slim for GTX 1650

#RUN apt-get update && apt-get install -y \
    #python3.10 python3-pip \
    #libglib2.0-0 libsm6 libxext6 libxrender-dev \
  #&& rm -rf /var/lib/apt/lists/*

#WORKDIR /app
#COPY requirements.txt ./
#RUN pip install --no-cache-dir -r requirements.txt

#COPY algorithm.py ./

#ENTRYPOINT ["python3", "algorithm.py"]


#FROM python:3.10-slim

#RUN apt-get update && apt-get install -y \
    #libglib2.0-0 \
    #libsm6 \
    #libxext6 \
    ##libxrender1 \
    #libgl1-mesa-glx \
    #libgl1-mesa-dev \
  #&& rm -rf /var/lib/apt/lists/*


#WORKDIR /usr/src/app
##COPY app/requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
#COPY app/algorithm.py .

#ENTRYPOINT ["python3", "algorithm.py"]


#FROM python:3.10-slim CPU
#FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

#----------------------------------------------------------------------
     # because I have GTX 1650 :(
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

#RUN apt-get update && apt-get install -y \
RUN apt-get update && \
    apt-get install -y -qq \
      python3.10 python3-pip \
      libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

COPY algorithm.py .

ENTRYPOINT ["python3", "algorithm.py"]


#-------------------------------RTX--------------------------------------------

#FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04


#RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    #python3.10 python3-pip \
    #libglib2.0-0 libsm6 libxext6 libxrender-dev \
  #&& rm -rf /var/lib/apt/lists/*


#WORKDIR /usr/src/app


#COPY requirements.txt .
#RUN python3 -m pip install --upgrade pip && \
    #python3 -m pip install --no-cache-dir \
      #torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    #python3 -m pip install --no-cache-dir -r requirements.txt


#COPY algorithm.py .


#ENTRYPOINT ["python3", "algorithm.py"]






#-----------RUNNING THE CONTAINER----------------
  # cd project-root\app
  #docker build -t denizbaryak_gorev3_2_3_2 .
  #docker run --rm --gpus all -v "${PWD}:/app" -v "${env:USERPROFILE}\.cache\torch:/root/.cache/torch" denizbaryak_gorev3_2_3_2
  #docker run --rm -v "${PWD}":/app denizbaryak_gorev3_2_3_2
#------------------------------------------------





