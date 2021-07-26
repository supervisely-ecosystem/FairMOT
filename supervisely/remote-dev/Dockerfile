
#  TODO: add DCN compile 


ARG IMAGE
FROM $IMAGE

RUN apt-get update && apt-get install -y openssh-server
EXPOSE 22

RUN apt-get install -y sudo
RUN mkdir -p /run/sshd


RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh


RUN bash Anaconda3-2021.05-Linux-x86_64.sh -b -p /miniconda
ENV PATH=$PATH:/miniconda/condabin:/miniconda/bin

RUN conda --version

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


RUN ls *

RUN git clone -b develop https://github.com/supervisely-ecosystem/FairMOT 

# RUN cd /FairMOT/ && git pull && git checkout develop && git pull
# RUN mkdir /FairMOT/data && cd /FairMOT/data && wget https://motchallenge.net/data/MOT20.zip && unzip MOT20.zip
# RUN rm -f -r /FairMOT/data/MOT20.zip && cd /FairMOT/data/MOT20/ && mkdir -p labels_with_ids/train && mkdir images && mkdir -p labels_with_ids/train && mv train images && mv test images


RUN add-apt-repository universe
RUN apt-get update
# RUN apt-get remove -y python3 python3-pip
# RUN apt-get install -y python3.8 python3-pip
# RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN python --version

# RUN pip install --upgrade pip

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
RUN pip install ffmpeg-python==0.2.0

# to support pandas.read_excel
RUN pip install xlrd
RUN pip install google-cloud-storage
RUN pip install python-slugify
RUN pip install psutil==5.4.5
RUN pip install cython

RUN apt-get update && apt-get install -y ffmpeg
RUN apt-get install -y tmux
RUN pip install protobuf==3.14.0
RUN pip install Werkzeug==1.0.1
RUN pip install sk-video==1.1.10
RUN pip install -U scikit-learn==0.24.2

RUN conda --version

RUN pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html


# ARG dataset_link=http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/238/o/0/rh/OZbarBya9HwaR335PexrfEquxj7uJbrsyJzXHiCehP1ZeWSSJWmXTcyIBEsEPj4wQGIZFecxq0fgD3wcXcUdJm3SnYDvpxS4CaHe3vIvefTgsuMHQt6xVmqHBPyA.tar

# ARG dataset_link=http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/238/M/b/yb/vxFdkNkOia8s7Xo5VBDHaNl3k3ftFuQaGvyn2pVXvMfHw9B21qgWkejxDeFr6f1E42ZKjvJJfosiwRtfjeuf1BmdlLYxFcHW057DYVmxkOx7cQsNfzhvbCw8WNf2.tar

# ARG dataset_link=http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/238/9/6/bP/8MEm7TUdWWqyDAKwHl9gFoDzPh3yoTIAVdwnP2cSbTFBpUVAu1snDpQ3X05j9HsGI9fhIrK0reNTXgMLixS7wv0HdlSzcNuCK5rcDOc0aIYh1JLoz5u9nvRlPy7Q.tar

# ARG dataset_link=http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/238/X/O/PB/BYf09klvmpx470idDYasYhckPXAfIX5r2lTRPboHNsr9OxBnDwnEFofeVOkQLZmMP1t5Krjyra3RLkFCnaKzx5bPXTWnMVtTqbdIMCg7pFY0ipzAKPRB22vDOUyP.tar

# ARG dataset_link=http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/238/Z/D/kL/OurtZ5velR1ZKtB2KPJRaQEORf5k3gCb2qPj2VefAPvmXPpn8ImkTwahyM6TyBUCLytj80YoB0nud8RhjFDirykdTPfbDqeVxSQfPh5QYhBYAPq47Tbaf62hm68H.tar

ARG dataset_link=http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/238/C/g/AH/4RIlv2OcvxxLGZfl9MsXbw6ZQsQw3BkuQbftyeG1tyFVVl7wiesnWuBr2E7Ce2Mk9WJuSJfEYDpuK0b4ORgFJZbWc4noNAUcuyGYsbUpNDGCNVCIKyxebdGdzghC.tar

RUN mkdir /FairMOT/data && cd /FairMOT/data &&  wget $dataset_link && tar -xvf *.tar && mv ds_* SLY_MOT
RUN ls /FairMOT/data
RUN rm -f -r /FairMOT/data/*.tar && cd /FairMOT/data/SLY_MOT/ && mkdir -p labels_with_ids/train && mkdir images && mkdir -p labels_with_ids/train && cp -r train images/test && mv train images

ARG ls

RUN cd /FairMOT/ && git pull

RUN pip install -r /FairMOT/requirements.txt 
# RUN python /FairMOT/src/rename_img_seq.py
RUN python /FairMOT/src/gen_labels.py
RUN cd /FairMOT/src/ && python ./gen_data_path.py
 


RUN git clone -b pytorch_1.7 https://github.com/ifzhang/DCNv2.git
#ENV PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:${PATH}"
#RUN nvidia-smi

RUN ls /usr/local/
RUN ls /usr/local/cuda

#ENV PYTHONUNBUFFERED=1
#ENV PATH="/usr/local/cuda-11.1/bin:${PATH}"
#ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

#RUN cd DCNv2 && ./make.sh
# RUN cd DCNv2 && CUDA_VISIBLE_DEVICES=0 ./make.sh





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

