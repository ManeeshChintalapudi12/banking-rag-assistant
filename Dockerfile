FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data/sample_docs ./data/sample_docs

# Build the vector index at image build time so the container is
# ready to serve queries immediately on startup.
RUN python -m app.build_index

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
