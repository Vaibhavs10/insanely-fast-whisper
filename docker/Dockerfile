FROM python:3.8
WORKDIR /app
COPY . /app
COPY ./docker/requirements.txt  requirements.txt
RUN pip install -r requirements.txt
ENTRYPOINT ["python3", "../src/insanely_fast_whisper/cli.py"]