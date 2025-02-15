FROM python:3.9-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install MLflow first explicitly
RUN pip install mlflow==2.3.0 && \
    pip install --no-cache-dir -r requirements.txt

# Install NLTK data
RUN python -m nltk.downloader punkt stopwords

# Create necessary directories
RUN mkdir -p models

# Copy the model and MLflow data
COPY models/spam_classifier.joblib models/
COPY mlruns mlruns/
COPY . .

# Expose ports for both FastAPI and MLflow
EXPOSE 8000
EXPOSE 5000

# Create start script
RUN echo '#!/bin/bash\n\
python -m mlflow ui --host 0.0.0.0 --port 5000 &\n\
uvicorn api.app:app --host 0.0.0.0 --port 8000\n'\
> /app/start.sh

RUN chmod +x /app/start.sh

CMD ["/bin/bash", "/app/start.sh"]