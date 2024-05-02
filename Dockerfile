FROM python:3.9
RUN pip install tensorflow==2.11.0 tensorflow_datasets==4.7.0
COPY model-prediction.py /
COPY model-selection.py /
COPY models.py /
COPY pipeline.py /