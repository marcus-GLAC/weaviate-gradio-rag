# Ứng dụng RAG với Weaviate, Gradio và Gemini/OpenAI

Đây là ứng dụng RAG (Retrieval Augmented Generation) cho phép bạn import file PDF và TXT vào và đặt câu hỏi về nội dung của chúng.

## Tính năng

- Import file PDF và TXT
- Tìm kiếm thông tin trong file đã import
- Chat RAG với sự hỗ trợ của AI (Gemini hoặc OpenAI)
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
   ```
4. Khởi động Docker containers:
   ```
   docker-compose up -d
   ```
5. Chạy ứng dụng:
   ```
   pip install -r requirements.txt
   python app.py
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