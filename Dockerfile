FROM gcr.io/deeplearning-platform-release/tf-cpu.1-14

ENV PYTHONHASHSEED "1"
ENV CUDA_VISIBLE_DEVICES "-1"
ENV TF_CUDNN_USE_AUTOTUNE "0"

RUN pip install --upgrade pip
RUN pip uninstall -y Pillow

ADD requirements.txt /usr/src/app/requirements.txt
RUN pip install -r /usr/src/app/requirements.txt

ADD conf/.theanorc /root/.theanorc
ADD conf/.theanorc /.theanorc

ADD . /usr/src/app

WORKDIR /usr/src/app