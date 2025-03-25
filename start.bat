@echo off
echo Kiểm tra xem Docker đã được cài đặt chưa...
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo Docker không được tìm thấy. Vui lòng cài đặt Docker trước.
    pause
    exit /b 1
)

echo Kiểm tra xem Docker Compose đã được cài đặt chưa...
where docker-compose >nul 2>nul
if %errorlevel% neq 0 (
    echo Docker Compose không được tìm thấy. Vui lòng cài đặt Docker Compose trước.
    pause
    exit /b 1
)

rem Kiểm tra xem file .env đã được tạo chưa
if not exist .env (
    echo File .env không tồn tại. Tạo từ .env.example...
    copy .env.example .env
    echo Vui lòng chỉnh sửa file .env và thêm API keys của bạn.
    pause
    exit /b 1
)

rem Khởi động Docker containers
echo Khởi động Weaviate và mô hình embedding...
docker-compose up -d

rem Đợi Weaviate khởi động
echo Đợi Weaviate và mô hình embedding khởi động...
timeout /t 15 /nobreak

rem Kiểm tra kết nối Weaviate
echo Kiểm tra kết nối Weaviate...
curl -s http://localhost:8080/v1/meta >nul
if %errorlevel% neq 0 (
    echo Không thể kết nối với Weaviate. Vui lòng kiểm tra logs:
    docker-compose logs
    pause
    exit /b 1
)

rem Cài đặt các phụ thuộc Python
echo Cài đặt các phụ thuộc Python...
pip install -r requirements.txt

rem Chạy ứng dụng
echo Khởi động ứng dụng RAG...
python app.py 