FROM debian:11

RUN apt -y update && apt -y upgrade

# install CatPreyAnalyzer fork adapted to use opencv cam and working on bullseye
# (https://github.com/richteas75/Cat_Prey_Analyzer)

# first install opencv and tensorflow (see https://github.com/richteas75/Cat_Prey_Analyzer and https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi)

RUN apt -y install git \
                   wget \
                   unzip \
                   vim

RUN apt -y install cython3 \
                   python-tk \
                   libjpeg-dev \
                   libtiff5-dev \
                   libpng-dev \
                   libavcodec-dev \
                   libavformat-dev \
                   libswscale-dev \
                   libv4l-dev \
                   libxvidcore-dev \
                   libx264-dev \
                   libatlas-base-dev \
                   python3-python-telegram-bot \
                   python3-tz \
                   python3-pil \
                   python3-lxml \
                   python3-opencv \
                   python3-pip

RUN pip3 install tensorflow \
                 tensorflow-io \
                 pillow \
                 lxml \
                 jupyter \
                 matplotlib

ARG PROTOC_ZIP=/tmp/protoc.zip
RUN wget --output-document "$PROTOC_ZIP" https://github.com/protocolbuffers/protobuf/releases/download/v3.20.3/protoc-3.20.3-linux-x86_64.zip && \
    unzip $PROTOC_ZIP -d /tmp && \
    mv /tmp/bin/protoc /usr/bin/protoc

RUN useradd --create-home cat
ARG CAT_HOME=/home/cat
WORKDIR $CAT_HOME

RUN mkdir tensorflow && \
	cd tensorflow && \
	git clone --depth 1 https://github.com/tensorflow/models.git

ARG RESEARCH_PATH=$CAT_HOME/tensorflow/models/research
ENV PYTHONPATH=$PYTHONPATH:$RESEARCH_PATH:$RESEARCH_PATH/slim

RUN cd $RESEARCH_PATH && protoc object_detection/protos/*.proto --python_out=.
RUN cd $RESEARCH_PATH/object_detection && \
	wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz && \
	tar -xzvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz

COPY . $CAT_HOME/CatPreyAnalyzer/
RUN chown --recursive cat:cat $CAT_HOME
RUN chmod +x $CAT_HOME/CatPreyAnalyzer/catCam_starter.sh

USER cat

ENTRYPOINT $HOME/CatPreyAnalyzer/catCam_starter.sh