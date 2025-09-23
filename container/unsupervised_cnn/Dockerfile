FROM python:3.10-slim

RUN pip install --no-cache-dir pandas scikit-learn numpy

WORKDIR /app
COPY unsupervised_cnn.py .

ENTRYPOINT ["python", "unsupervised_cnn.py"]
