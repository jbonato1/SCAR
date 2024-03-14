FROM nvidia/cuda:11.7.1-base-ubuntu22.04
COPY requirements.txt .
RUN  apt-get update 
RUN  apt-get install -y
RUN  apt install git -y
RUN  apt install -y python3.10 
RUN  apt install -y python3-pip
RUN  apt-get install ffmpeg libsm6 libxext6  -y
RUN  pip3 install --no-cache-dir -r requirements.txt
RUN alias python=python3