@echo off
setlocal enabledelayedexpansion

REM Kiểm tra Docker
where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Docker không được cài đặt. Vui lòng cài đặt Docker trước khi tiếp tục.
    exit /b 1
)

REM Kiểm tra Docker đang chạy
docker info >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Docker không đang chạy. Vui lòng khởi động Docker trước khi tiếp tục.
    exit /b 1
)

REM Cài đặt các thư viện Python
echo Cài đặt các thư viện Python...
pip install -r requirements.txt

REM Kiểm tra và chạy container embedding model
docker ps | findstr "embedding-model" >nul
if %ERRORLEVEL% neq 0 (
    docker ps -a | findstr "embedding-model" >nul
    if %ERRORLEVEL% equ 0 (
        echo Khởi động lại container embedding-model...
        docker start embedding-model
    ) else (
        echo Tạo và chạy container embedding-model...
        docker run -d -p 8000:8080 --name embedding-model semitechnologies/transformers-inference:sentence-transformers-multi-qa-MiniLM-L6-cos-v1
    )
) else (
    echo Container embedding-model đã đang chạy.
)

REM Kiểm tra file .env
if not exist .env (
    if exist .env.example (
        echo Tạo file .env từ .env.example...
        copy .env.example .env
        echo Vui lòng chỉnh sửa file .env để thêm API key nếu bạn muốn sử dụng tính năng LLM.
    )
)

REM Hỏi người dùng muốn chạy phiên bản nào
echo Chọn phiên bản để chạy:
echo 1. Phiên bản cơ bản (không có LLM)
echo 2. Phiên bản đầy đủ (có tích hợp LLM)
set /p choice=Nhập lựa chọn của bạn (1 hoặc 2): 

if "%choice%"=="1" (
    echo Chạy phiên bản cơ bản...
    python weaviate_rag_app.py
) else if "%choice%"=="2" (
    echo Chạy phiên bản đầy đủ với LLM...
    python weaviate_rag_app_with_llm.py
) else (
    echo Lựa chọn không hợp lệ. Chạy phiên bản cơ bản...
    python weaviate_rag_app.py
)
