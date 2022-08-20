FROM nvidia/cuda:11.4.0-runtime-ubuntu18.04
WORKDIR /usr/project

RUN apt-get clean && apt-get update -y -qq
# Install generic packages
RUN apt-get install -y \
        build-essential \
        cmake \
        pkg-config \
        wget \
        swig \
        git \
        curl \
        unzip \
        libaio1 \
        nano \
        freetds-dev \
        unixodbc \
        unixodbc-dev \
        libjpeg-dev \
        libtiff5-dev \
        libpng-dev \
        libgtk2.0-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libatlas-base-dev \
        gfortran \
        libhdf5-dev \
        libtbb2 \
        libtbb-dev \
        libgl1-mesa-glx \
        openmpi-bin

# Install python
RUN apt-get install -y python3 python3-pip python3-dev python3-setuptools
RUN rm /usr/bin/python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install pip
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py
RUN python get-pip.py
RUN pip install --upgrade pip && rm get-pip.py

# Install python libraries
RUN pip install --default-timeout=1000 tensorflow
RUN pip install --default-timeout=1000 pandas \
            numpy \
            scikit-learn \
            scipy \
            matplotlib \
            seaborn \
            opencv-python \
            Pillow \
            jupyter \
            tqdm \
            tensorboard \
            jupyter_contrib_nbextensions

RUN jupyter contrib nbextension install --user

RUN rm -rf ~/.cache/pip
RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#ENTRYPOINT [ "tensorboard", "--logdir", " experiments/tensorboard", "--host", "0.0.0.0", "&" ]
