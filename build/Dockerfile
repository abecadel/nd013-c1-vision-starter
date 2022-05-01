FROM nvcr.io/nvidia/tensorflow:22.04-tf2-py3


# Let us install tzdata painlessly
ENV DEBIAN_FRONTEND=noninteractive

# Needed for string substitution
SHELL ["/bin/bash", "-c"]


RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        git-core \
        g++ \
        vim \
        zip \
        zlib1g-dev \
        curl \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        libcairo2-dev \
        pkg-config \
        software-properties-common \
        unzip \
        wget \ 
        protobuf-compiler \
        python3-dev


WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

# ENV TF_CPP_MIN_LOG_LEVEL=2

# get models
RUN git clone https://github.com/tensorflow/models.git && \
    cd /app/models/research/ && \
    protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf2/setup.py . && \
    python -m pip install .

# Install google cloud SDK
RUN curl -sSL https://sdk.cloud.google.com > /tmp/gcl && bash /tmp/gcl --install-dir=~/gcloud --disable-prompts
ENV PATH="$PATH:/root/gcloud/google-cloud-sdk/bin"

# install waymo dataset
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64 && \
    chmod o+x bazelisk-linux-amd64 && \
    mv bazelisk-linux-amd64 /usr/local/bin/bazel 

#ENV USE_BAZEL_VERSION=3.1.0
#RUN git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od && \
#    cd waymo-od && \
#    git checkout remotes/origin/master && \
#    bash configure.sh && \
#    bazel build

# get pretrained model
RUN mkdir -p pretrained_model && \ 
    cd pretrained_model && \
    wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz && \
    tar xvzf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz


