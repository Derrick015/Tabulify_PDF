FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./pyproject.toml
COPY src/ ./src/

RUN pip install --no-cache-dir .

COPY app.py .
COPY logo.png .
COPY favicon.png .

RUN mkdir -p logs output_files

EXPOSE 8501

CMD ["sh", "-c", "streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true"]