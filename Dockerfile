
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --timeout 300 --retries 5 -r requirements.txt

COPY app.py predict_word.py segment_word.py utils.py character_map.json ./

COPY models/model.keras ./models/model.keras

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
