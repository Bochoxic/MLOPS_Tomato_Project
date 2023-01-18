# Base image
FROM python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN pip install wandb

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY reports/ reports/
COPY models/ models/
COPY config/ config/
COPY .dvc/config .dvc/config
COPY data.dvc data.dvc

WORKDIR /
RUN pip install dvc[gs]
RUN dvc config core.no_scm true
RUN dvc pull
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]