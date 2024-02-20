FROM nvidia/cuda:11.7.1-base-ubuntu20.04

ARG PORT
ARG IP_ADDRESS="0.0.0.0"
ENV PORT ${PORT}
ENV IP_ADDRESS ${IP_ADDRESS}

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install --yes --no-install-recommends \
        git \
        less \
        python-is-python3 \
        python3.9 \
        python3-pip \
        tree \
        vim \
        wget \
	&& \
    rm -rf /var/lib/apt/lists/*

## Install and activate miniconda
RUN mkdir -p opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.11.0-2-Linux-x86_64.sh -O /opt/conda/miniconda.sh \
    && bash /opt/conda/miniconda.sh -b -p /opt/miniconda
RUN . /opt/miniconda/bin/activate 

## Map conda packages to system
RUN ln -sf /opt/miniconda/bin/python /usr/bin/python
RUN ln -sf /opt/miniconda/bin/pip /usr/bin/pip
RUN ln -sf /opt/miniconda/bin/conda /usr/bin/conda

## Define work directory and install clockwork package
WORKDIR /app
COPY . /app
RUN pip install -r docker/requirements.txt --no-cache-dir
RUN pip install . --no-cache-dir

## Run the test suite
RUN python -m pytest tests

## Run jupyter notebook
CMD ["sh", "-c", "python -m notebook notebooks --port ${PORT} --no-browser --ip ${IP_ADDRESS} --allow-root"]
