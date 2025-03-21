# Hướng dẫn nhanh

## Windows

1. Đảm bảo Docker đã được cài đặt và đang chạy
2. Chạy file `setup_and_run.bat` bằng cách nhấp đúp vào nó
3. Làm theo các hướng dẫn trên màn hình

## Linux/Mac

1. Đảm bảo Docker đã được cài đặt và đang chạy
2. Mở terminal trong thư mục này
3. Cấp quyền thực thi cho script: `chmod +x setup_and_run.sh`
4. Chạy script: `./setup_and_run.sh`
5. Làm theo các hướng dẫn trên màn hình

## Thử nghiệm nhanh

1. Sau khi ứng dụng đã chạy, mở trình duyệt và truy cập: http://localhost:7860
2. Trong tab "Import Dữ Liệu", upload file mẫu từ thư mục `sample_data`
3. Chuyển sang tab "Tìm Kiếm" và thử tìm kiếm với các từ khóa như:
   - "Weaviate là gì"
   - "RAG hoạt động như thế nào"
   - "Ứng dụng của Weaviate"

## Lưu ý

- Nếu bạn muốn sử dụng tính năng RAG với LLM, hãy đảm bảo đã thiết lập API key trong file `.env`
- Dữ liệu được lưu trong thư mục `data`, bạn có thể sao lưu thư mục này để giữ lại dữ liệu
