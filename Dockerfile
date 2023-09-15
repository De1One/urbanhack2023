FROM ubuntu:22.04

WORKDIR /solution
COPY . .

# dependencies
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y \
        build-essential git python3 python3-pip wget \
        ffmpeg libsm6 libxext6 libxrender1 libglib2.0-0

RUN pip3 install -U pip
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN mim install mmcv==2.0.0 # can be deleted

# model weights
RUN mkdir -p ./weights
COPY weights/faster-rcnn-best.pth ./weights

# input and output folders
RUN mkdir -p ./private/images
RUN mkdir -p ./private/labels
RUN mkdir -p ./output

# !!!! ONLY FOR THE TEST RUN - DELETE BEFORE SUBMITTING --->>>
# COPY images ./private/images
# COPY labels ./private/labels
# <<<---

CMD /bin/sh -c "python3 solution.py && python3 scorer.py"
