ARG IMAGE
FROM $IMAGE

RUN apt-get update && apt-get install -y openssh-server
EXPOSE 22

RUN apt-get install -y sudo
RUN mkdir -p /run/sshd



RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        nano \
        unzip \
        ca-certificates\
        libjpeg-dev \
        libpng-dev \
        software-properties-common \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log


RUN cd /

RUN git clone https://github.com/ifzhang/FairMOT

RUN cd /FairMOT/
RUN git checkout develop && git pull
RUN mkdir data && cd data
RUN wget https://motchallenge.net/data/MOT20.zip
RUN unzip MOT20
RUN rm -f -r MOT20.zip && cd MOT20/
RUN mkdir images && mkdir -p labels_with_ids/train
RUN mv test images && mv train images


RUN add-apt-repository universe
RUN apt-get update
RUN apt-get install -y python3.8 python3-pip
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN pip install --upgrade pip

RUN pip install \
        python-json-logger==0.1.11 \
        pybase64==1.0.2 \
        shapely==1.7.1 \
        imgaug==0.4.0 \
        numpy==1.19 \
        opencv-python==3.4.11.39 \
        scipy==1.5.2 \
        scikit-image==0.17.1 \
        matplotlib==3.3.2 \
        pillow \
        requests==2.24.0 \
        networkx==2.5 \
        jsonschema==2.6.0

RUN pip install \
    pandas==1.1.3 \
    grpcio==1.32.0 \
    grpcio-tools==1.32.0


RUN apt-get update && apt-get install -y openssh-server
RUN mkdir -p /var/run/sshd

RUN pip install plotly==4.11.0
RUN pip install docker==4.4.1
RUN pip install fuzzywuzzy

# to support pandas.read_excel
RUN pip install xlrd
RUN pip install google-cloud-storage
RUN pip install python-slugify
RUN pip install psutil==5.4.5
RUN pip install cython

RUN apt-get update && apt-get install -y ffmpeg
RUN pip install protobuf==3.14.0
RUN pip install Werkzeug==1.0.1
RUN pip install sk-video==1.1.10


RUN pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# RUN export CUDA_HOME=/local/usr/cuda-11.1

# RUN git clone https://github.com/ifzhang/DCNv2.git
# RUN cd DCNv2 && ./make.sh

RUN python /FairMOT/src/gen_labels_20.py 
RUN pip install -r /FairMOT/requirements.txt 




# MORE SECURE: use a trusted RSA key
ARG home=/root
RUN mkdir $home/.ssh
COPY authorized_keys $home/.ssh/authorized_keys
#COPY id_rsa.pub $home/.ssh/authorized_keys
RUN chown root:root $home/.ssh/authorized_keys && \
    chmod 600 $home/.ssh/authorized_keys

COPY sshd_deamon.sh /sshd_deamon.sh
RUN chmod 755 /sshd_deamon.sh
CMD ["/sshd_deamon.sh"]
ENTRYPOINT ["sh", "-c", "/sshd_deamon.sh"]

