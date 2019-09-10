FROM python:3

ADD requirements.txt /usr/src/app/requirements.txt
RUN pip install -r /usr/src/app/requirements.txt

ADD . /usr/src/app

WORKDIR /usr/src/app

EXPOSE 8080

VOLUME /usr/src/app/data

CMD ["python", "/usr/src/app/api.py"]