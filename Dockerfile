FROM tensorflow/tensorflow:1.12.0

RUN apt-get -y update
RUN apt-get -y install pypy samtools wget pypy-dev parallel
WORKDIR /opt/clairvoyante
COPY . .
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN pypy get-pip.py
RUN python -m pip install -r requirements.txt
RUN pypy -m pip install -r requirements_pypy.txt

WORKDIR /data
