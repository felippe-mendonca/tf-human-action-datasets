FROM tensorflow/tensorflow:1.12.0-py3

ADD . /devel
WORKDIR /devel
RUN pip install -r requirements.txt

ENTRYPOINT [ "bash" ]
