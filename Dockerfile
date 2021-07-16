FROM ubuntu:bionic

FROM python:3.6-slim

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y vim && \ 
    apt-get install -y ffmpeg libsm6 libxext6 libxrender-dev && \ 
    apt-get install -y g++ 

RUN apt-get install -y libgdal-dev

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal

ENV C_INCLUDE_PATH=/usr/include/gdal

RUN pip3 install --upgrade pip

RUN pip3 install numpy matplotlib scipy pandas scikit-learn==0.19.0 opencv-python 

RUN pip3 install GDAL==2.2.3

ENTRYPOINT ["python", "shifty.py"]

CMD ["-h"]
