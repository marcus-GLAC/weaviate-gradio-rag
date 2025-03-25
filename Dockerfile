FROM python:3.10-slim

WORKDIR /app

# Cài đặt các công cụ cần thiết
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Sao chép requirements file
COPY requirements.txt .

# Cài đặt dependencies
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Tạo thư mục uploads
RUN mkdir -p /app/uploads

# Sao chép code
COPY app.py .
COPY .env .

# Expose port 7860 cho Gradio
EXPOSE 7860

# Đặt các biến môi trường
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_ROOT_PATH=/
ENV GRADIO_ALLOWED_ORIGINS=*
ENV PYTHONUNBUFFERED=1

# Chạy ứng dụng Gradio
CMD ["python", "app.py"] 