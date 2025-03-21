# Ứng dụng RAG với Weaviate và Gradio

Ứng dụng này sử dụng Weaviate làm vector database và Gradio để tạo giao diện người dùng, cho phép import và tìm kiếm thông tin từ file .txt và PDF.

## Cài đặt

### Yêu cầu

- Python 3.8+
- Docker

### Bước 1: Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

### Bước 2: Thiết lập Docker cho mô hình embedding

```bash
docker run -d -p 8000:8080 --name embedding-model semitechnologies/transformers-inference:sentence-transformers-multi-qa-MiniLM-L6-cos-v1
```

### Bước 3: Thiết lập OpenAI API Key (tùy chọn, chỉ cần nếu sử dụng tính năng RAG với LLM)

Sao chép file `.env.example` thành `.env` và thêm OpenAI API key của bạn:

```bash
cp .env.example .env
```

Sau đó chỉnh sửa file `.env` và thêm API key của bạn.

## Chạy ứng dụng

### Phiên bản cơ bản (không có LLM)

```bash
python weaviate_rag_app.py
```

### Phiên bản đầy đủ (có tích hợp LLM)

```bash
python weaviate_rag_app_with_llm.py
```

Sau khi chạy, ứng dụng sẽ khả dụng tại địa chỉ: http://localhost:7860

## Sử dụng ứng dụng

### 1. Import dữ liệu

- Chuyển đến tab "Import Dữ Liệu"
- Upload file .txt hoặc .pdf
- Nhấn nút "Import"

### 2. Tìm kiếm thông tin

- Chuyển đến tab "Tìm Kiếm"
- Nhập câu hỏi hoặc từ khóa tìm kiếm
- Điều chỉnh số lượng kết quả hiển thị (nếu cần)
- Nhấn nút "Tìm kiếm"

### 3. Sử dụng RAG với LLM (chỉ có trong phiên bản đầy đủ)

- Chuyển đến tab "RAG Q&A"
- Nhập câu hỏi của bạn
- Nhấn nút "Trả lời"
- Hệ thống sẽ sử dụng LLM để tạo câu trả lời dựa trên dữ liệu đã import

## Cấu trúc thư mục

```
weaviate-gradio-rag/
├── weaviate_rag_app.py         # Ứng dụng cơ bản
├── weaviate_rag_app_with_llm.py # Ứng dụng đầy đủ với LLM
├── requirements.txt            # Danh sách thư viện cần thiết
├── .env.example                # Mẫu file cấu hình biến môi trường
├── data/                       # Thư mục lưu dữ liệu Weaviate
└── backups/                    # Thư mục backup
```

## Khắc phục sự cố

### 1. Lỗi kết nối đến mô hình embedding

Kiểm tra container embedding-model đã chạy chưa:

```bash
docker ps
```

Nếu container không chạy, khởi động lại:

```bash
docker start embedding-model
```

### 2. Lỗi khi sử dụng LLM

Kiểm tra file `.env` đã có API key chưa và API key có hợp lệ không.

### 3. Xóa dữ liệu và bắt đầu lại

```bash
# Dừng container
docker stop embedding-model

# Xóa thư mục dữ liệu
rm -rf ./data

# Khởi động lại container
docker start embedding-model
