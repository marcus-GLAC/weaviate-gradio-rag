#!/bin/bash

# Kiểm tra Docker
if ! command -v docker &> /dev/null; then
    echo "Docker không được cài đặt. Vui lòng cài đặt Docker trước khi tiếp tục."
    exit 1
fi

# Kiểm tra Docker đang chạy
if ! docker info &> /dev/null; then
    echo "Docker không đang chạy. Vui lòng khởi động Docker trước khi tiếp tục."
    exit 1
fi

# Cài đặt các thư viện Python
echo "Cài đặt các thư viện Python..."
pip install -r requirements.txt

# Kiểm tra và chạy container embedding model
if ! docker ps | grep -q embedding-model; then
    if docker ps -a | grep -q embedding-model; then
        echo "Khởi động lại container embedding-model..."
        docker start embedding-model
    else
        echo "Tạo và chạy container embedding-model..."
        docker run -d -p 8000:8080 --name embedding-model semitechnologies/transformers-inference:sentence-transformers-multi-qa-MiniLM-L6-cos-v1
    fi
else
    echo "Container embedding-model đã đang chạy."
fi

# Kiểm tra file .env
if [ ! -f .env ] && [ -f .env.example ]; then
    echo "Tạo file .env từ .env.example..."
    cp .env.example .env
    echo "Vui lòng chỉnh sửa file .env để thêm API key nếu bạn muốn sử dụng tính năng LLM."
fi

# Hỏi người dùng muốn chạy phiên bản nào
echo "Chọn phiên bản để chạy:"
echo "1. Phiên bản cơ bản (không có LLM)"
echo "2. Phiên bản đầy đủ (có tích hợp LLM)"
read -p "Nhập lựa chọn của bạn (1 hoặc 2): " choice

case $choice in
    1)
        echo "Chạy phiên bản cơ bản..."
        python weaviate_rag_app.py
        ;;
    2)
        echo "Chạy phiên bản đầy đủ với LLM..."
        python weaviate_rag_app_with_llm.py
        ;;
    *)
        echo "Lựa chọn không hợp lệ. Chạy phiên bản cơ bản..."
        python weaviate_rag_app.py
        ;;
esac
