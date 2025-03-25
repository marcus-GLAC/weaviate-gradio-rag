#!/bin/bash

# Kiểm tra xem docker và docker-compose đã được cài đặt chưa
if ! command -v docker &> /dev/null; then
    echo "Docker không được tìm thấy. Vui lòng cài đặt Docker trước."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose không được tìm thấy. Vui lòng cài đặt Docker Compose trước."
    exit 1
fi

# Kiểm tra xem file .env đã được tạo chưa
if [ ! -f .env ]; then
    echo "File .env không tồn tại. Tạo từ .env.example..."
    cp .env.example .env
    echo "Vui lòng chỉnh sửa file .env và thêm API keys của bạn."
    exit 1
fi

# Khởi động Docker containers
echo "Khởi động Weaviate và mô hình embedding..."
docker-compose up -d

# Đợi Weaviate khởi động
echo "Đợi Weaviate và mô hình embedding khởi động..."
sleep 15

# Kiểm tra kết nối Weaviate
echo "Kiểm tra kết nối Weaviate..."
curl -s http://localhost:8080/v1/meta > /dev/null
if [ $? -ne 0 ]; then
    echo "Không thể kết nối với Weaviate. Vui lòng kiểm tra logs:"
    docker-compose logs
    exit 1
fi

# Cài đặt các phụ thuộc Python
echo "Cài đặt các phụ thuộc Python..."
pip install -r requirements.txt

# Chạy ứng dụng
echo "Khởi động ứng dụng RAG..."
python app.py 