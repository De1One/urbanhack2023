FROM ubuntu:22.04

# dependencies
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y \
        build-essential git python3 python3-pip wget \
        ffmpeg libsm6 libxext6 libxrender1 libglib2.0-0

COPY requirements.txt .

RUN pip3 install -U pip
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN mim install mmcv==2.0.0 # can be deleted
RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

WORKDIR /solution
COPY . .

# model weights
RUN mkdir -p ./weights
COPY weights/model_final.pth ./weights

# input and output folders
RUN mkdir -p ./private/images
RUN mkdir -p ./private/labels
RUN mkdir -p ./output

CMD /bin/sh -c "python3 solution.py && python3 scorer.py"
