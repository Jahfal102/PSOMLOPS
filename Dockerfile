# Install the base requirements for the app.
# This stage is to support development.
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit" , "run", "app.py"]

