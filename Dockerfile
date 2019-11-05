FROM tensorflow/tensorflow:1.14.0-py3

RUN apt update && apt install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV PYTHONIOENCODING UTF-8

RUN apt install -y libsm6 libxext6 libxrender-dev libglib2.0-0 poppler-utils

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

ADD . /code
WORKDIR /code

ENV FTI_IDCARD_HOME=/code
ENV PYTHONPATH=/code

ENTRYPOINT gunicorn main:app \
            --bind 0.0.0.0:5000  \
            --worker-class sanic.worker.GunicornWorker \
            --timeout 120 \
            --threads 4 \
            --workers=${FTI_IDCARD_WORKERS}
