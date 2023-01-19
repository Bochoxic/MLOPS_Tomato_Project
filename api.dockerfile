FROM python:3.8-slim

EXPOSE $PORT

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install pydantic
RUN pip install uvicorn
RUN pip install python-multipart

COPY src/api/ src/api
COPY src/models/ src/models/
COPY models/ models/
COPY requirements_api.txt requirements_api.txt
COPY prueba.jpg prueba.jpg

RUN pip install -r requirements_api.txt

CMD ["python", "src/api/app/main.py"]