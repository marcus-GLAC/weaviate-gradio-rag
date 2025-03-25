# Ứng Dụng RAG Với Weaviate, Gradio, Gemini & OpenAI

Ứng dụng RAG (Retrieval Augmented Generation) cho phép bạn import file PDF và TXT, rồi đặt câu hỏi để truy xuất thông tin từ các tài liệu này.

## Các Tính Năng Chính

- Import file PDF và TXT vào vector database
- Tìm kiếm thông tin trong các tài liệu đã import
- Chat với nội dung tài liệu sử dụng các mô hình AI mới nhất:
  - **Gemini**: gemini-2.0-flash
  - **OpenAI**: gpt-4o-mini, gpt-4o
- Giao diện người dùng đơn giản, thân thiện
- Hiển thị nguồn tham khảo cho mỗi câu trả lời

## Cài Đặt & Chạy

### Yêu Cầu
- Docker và Docker Compose
- API key cho Gemini (Google) hoặc OpenAI

### Các Bước Cài Đặt

1. **Chuẩn bị file cấu hình**
   ```bash
   cp .env.example .env
   ```
   
   Sau đó chỉnh sửa `.env` và thêm API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   DEFAULT_LLM_PROVIDER=gemini
   ```

2. **Khởi động ứng dụng**
   ```bash
   docker-compose up -d
   ```

3. **Truy cập ứng dụng**
   - Nếu chạy trên máy tính: http://localhost:7860
   - Nếu chạy trên hosting: http://<địa-chỉ-IP>:7860

## Sử Dụng Ứng Dụng

1. **Import Dữ Liệu**
   - Chọn tab "Import Dữ Liệu"
   - Upload file PDF hoặc TXT
   - Nhấn nút "Import"

2. **Tìm Kiếm**
   - Chọn tab "Tìm Kiếm"
   - Nhập câu hỏi và nhấn "Tìm kiếm"
   - Kết quả sẽ hiển thị các đoạn liên quan nhất

3. **Chat RAG**
   - Chọn tab "Chat RAG"
   - Chọn mô hình LLM (Gemini hoặc OpenAI)
   - Chọn mô hình cụ thể từ dropdown
   - Điều chỉnh temperature nếu cần
   - Nhập câu hỏi và nhận câu trả lời dựa trên nội dung đã import

## Khắc Phục Sự Cố

### Lỗi "Không thể kết nối Weaviate"

1. **Kiểm tra trạng thái các container**
   ```bash
   docker ps
   ```
   
   Tất cả các container (weaviate-db, embedding-model, rag-app) phải đang chạy.

2. **Kiểm tra logs**
   ```bash
   docker logs weaviate-db
   ```

3. **Khởi động lại các service**
   ```bash
   docker-compose down
   docker-compose up -d
   ```

4. **Đảm bảo mạng Docker đúng**
   ```bash
   docker network ls
   ```
   
   Nếu có lỗi kết nối giữa các container:
   ```bash
   docker-compose down
   docker network prune
   docker-compose up -d
   ```

5. **Kiểm tra cổng và tường lửa**
   Đảm bảo cổng 8080 không bị chặn và Weaviate có thể truy cập.

## Cấu Hình Trên Hosting

Nếu chạy trên hosting, cần đảm bảo:

1. Cổng 7860 và 8080 được mở:
   ```bash
   sudo ufw allow 7860/tcp
   sudo ufw allow 8080/tcp
   ```

2. Truy cập qua địa chỉ IP của hosting:
   ```
   http://<địa-chỉ-IP>:7860
   ```

## Cấu Trúc Dự Án

- `app.py` - File chính của ứng dụng
- `Dockerfile` - Cấu hình để build container
- `docker-compose.yml` - Định nghĩa các service
- `.env` - Cấu hình API keys và model
- `uploads/` - Thư mục chứa file tải lên

## Quản Lý Docker

- Xem logs: `docker-compose logs -f app`
- Khởi động lại: `docker-compose restart app`
- Dừng ứng dụng: `docker-compose down`