# Ứng dụng RAG với Weaviate, Gradio và Gemini/OpenAI

Đây là ứng dụng RAG (Retrieval Augmented Generation) cho phép bạn import file PDF và TXT vào và đặt câu hỏi về nội dung của chúng.

## Tính năng

- Import file PDF và TXT
- Tìm kiếm thông tin trong file đã import
- Chat RAG với sự hỗ trợ của AI (Gemini hoặc OpenAI)
- Lựa chọn mô hình cụ thể (GPT-3.5, GPT-4, Gemini Pro, Gemini 1.5 Pro)
- Điều chỉnh temperature của mô hình AI
- Giao diện thân thiện với người dùng thông qua Gradio

## Cài đặt

### Cài đặt bằng Docker (Khuyên dùng)

1. Đảm bảo bạn đã cài đặt Docker và Docker Compose.
2. Sao chép file `.env.example` thành `.env` và cập nhật API keys:
   ```
   cp .env.example .env
   ```
3. Chỉnh sửa file `.env` và thêm API keys của bạn:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   DEFAULT_LLM_PROVIDER=gemini  # hoặc openai
   
   # Cài đặt mô hình OpenAI (Tùy chọn)
   OPENAI_MODEL=gpt-3.5-turbo  # hoặc gpt-4, gpt-4-turbo
   OPENAI_TEMPERATURE=0
   
   # Cài đặt mô hình Gemini (Tùy chọn)
   GEMINI_MODEL=gemini-pro  # hoặc gemini-1.5-pro
   GEMINI_TEMPERATURE=0
   ```
4. Khởi động toàn bộ ứng dụng bằng Docker:
   ```
   docker-compose up -d
   ```
5. Truy cập ứng dụng:
   - Nếu chạy trên máy tính cá nhân: http://localhost:7860
   - Nếu chạy trên hosting/server: http://<địa-chỉ-IP-của-hosting>:7860
   - Nếu bạn có tên miền: http://<tên-miền>:7860 (sau khi đã cấu hình DNS)

Docker sẽ tự động tạo:
- Container cho mô hình embedding (transformers-inference)
- Container cho Weaviate database
- Container cho ứng dụng Gradio
- Volume để lưu trữ dữ liệu Weaviate
- Volume để lưu trữ các file được tải lên

### Các lệnh Docker hữu ích

- Xem logs của ứng dụng:
  ```
  docker-compose logs -f app
  ```

- Khởi động lại ứng dụng:
  ```
  docker-compose restart app
  ```

- Dừng toàn bộ ứng dụng:
  ```
  docker-compose down
  ```

- Dừng toàn bộ ứng dụng và xóa dữ liệu:
  ```
  docker-compose down -v
  ```

### Cài đặt thủ công

1. Sao chép file `.env.example` thành `.env` và cập nhật API keys.
2. Cài đặt Weaviate Embedded hoặc chạy Weaviate Server riêng biệt.
3. Cài đặt các phụ thuộc:
   ```
   pip install -r requirements.txt
   ```
4. Chạy ứng dụng:
   ```
   python app.py
   ```

## Cách sử dụng

1. Mở ứng dụng trong trình duyệt (mặc định là địa chỉ http://127.0.0.1:7860)
2. Tab "Import Dữ Liệu": Upload file PDF hoặc TXT để thêm vào cơ sở dữ liệu
3. Tab "Tìm Kiếm": Tìm kiếm thông tin trong các file đã import
4. Tab "Chat RAG": Đặt câu hỏi và nhận câu trả lời từ AI dựa trên nội dung của các file

## Cấu trúc mã nguồn

- `app.py`: File chính của ứng dụng
- `docker-compose.yml`: Cấu hình Docker cho Weaviate và mô hình embedding
- `.env`: Lưu trữ API keys cho Gemini và OpenAI

## Vấn đề thường gặp

- **Không thể kết nối Weaviate**: Đảm bảo rằng Weaviate đang chạy và có thể truy cập được qua http://localhost:8080
- **Lỗi API key**: Đảm bảo rằng bạn đã cung cấp đúng API key cho mô hình bạn chọn trong file `.env`

## Giấy phép

MIT License

### Cấu hình cho hosting

Nếu bạn triển khai trên hosting, hãy đảm bảo:

1. Mở cổng 7860 trong tường lửa của hosting
   ```bash
   # Ví dụ trên Ubuntu với UFW
   sudo ufw allow 7860/tcp
   ```

2. Nếu máy chủ sử dụng Firewalld:
   ```bash
   sudo firewall-cmd --permanent --add-port=7860/tcp
   sudo firewall-cmd --reload
   ```

3. Kiểm tra xem cổng đã được mở chưa:
   ```bash
   # Với netstat
   sudo netstat -tuln | grep 7860
   
   # Hoặc với ss
   sudo ss -tuln | grep 7860
   ```
   
4. Nếu muốn bảo mật hơn, bạn có thể cấu hình một reverse proxy (như Nginx) và SSL.