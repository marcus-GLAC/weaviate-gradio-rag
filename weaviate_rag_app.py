import os
import gradio as gr
import weaviate
from weaviate.embedded import EmbeddedOptions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
import tempfile

# Thiết lập Weaviate Embedded
embedded_options = EmbeddedOptions(
    additional_env_vars={
        "ENABLE_MODULES": "backup-filesystem,text2vec-transformers",
        "BACKUP_FILESYSTEM_PATH": "./backups",
        "LOG_LEVEL": "panic",
        "TRANSFORMERS_INFERENCE_API": "http://localhost:8000"
    },
    persistence_data_path="./data",
)

# Khởi tạo Weaviate client
client = weaviate.WeaviateClient(embedded_options=embedded_options)
client.connect()
print(f"Weaviate connection status: {client.is_ready()}")

# Tên collection
COLLECTION_NAME = "Documents"

# Tạo collection nếu chưa tồn tại
def create_collection():
    # Kiểm tra xem collection đã tồn tại chưa
    collections = client.collections.list_all()
    collection_names = [collection.name for collection in collections]
    
    if COLLECTION_NAME not in collection_names:
        # Tạo collection mới
        client.collections.create(
            name=COLLECTION_NAME,
            properties=[
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The content of the document chunk",
                },
                {
                    "name": "source",
                    "dataType": ["text"],
                    "description": "The source file of the document",
                },
                {
                    "name": "chunk_id",
                    "dataType": ["int"],
                    "description": "The chunk ID within the document",
                }
            ],
            vectorizer_config=weaviate.config.Configure.Vectorizer.text2vec_transformers(),
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists")

# Gọi hàm tạo collection
create_collection()

# Hàm xử lý và import file
def process_file(file_obj):
    # Lấy đường dẫn tạm thời của file đã upload
    temp_path = file_obj.name
    file_extension = os.path.splitext(temp_path)[1].lower()
    
    # Xử lý dựa trên loại file
    if file_extension == '.pdf':
        loader = PyPDFLoader(temp_path)
    elif file_extension == '.txt':
        loader = TextLoader(temp_path)
    else:
        return f"Không hỗ trợ định dạng file {file_extension}"
    
    # Load và chia nhỏ văn bản
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    # Lấy collection
    collection = client.collections.get(COLLECTION_NAME)
    
    # Import các chunk vào Weaviate
    source_filename = os.path.basename(temp_path)
    for i, chunk in enumerate(chunks):
        collection.data.insert({
            "content": chunk.page_content,
            "source": source_filename,
            "chunk_id": i
        })
    
    return f"Đã import thành công {len(chunks)} chunks từ file {source_filename}"

# Hàm tìm kiếm
def search_documents(query, top_k=5):
    if not query:
        return "Vui lòng nhập câu hỏi để tìm kiếm"
    
    collection = client.collections.get(COLLECTION_NAME)
    results = collection.query.near_text(
        query=query,
        limit=top_k
    )
    
    if not results.objects:
        return "Không tìm thấy kết quả phù hợp"
    
    formatted_results = []
    for i, obj in enumerate(results.objects):
        formatted_results.append(f"### Kết quả {i+1}\n")
        formatted_results.append(f"**Nguồn:** {obj.properties['source']}\n")
        formatted_results.append(f"**Nội dung:**\n{obj.properties['content']}\n\n")
    
    return "\n".join(formatted_results)

# Tạo giao diện Gradio
with gr.Blocks(title="RAG với Weaviate") as demo:
    gr.Markdown("# Ứng dụng RAG với Weaviate và Gradio")
    
    with gr.Tab("Import Dữ Liệu"):
        file_input = gr.File(label="Upload file (.txt hoặc .pdf)")
        import_button = gr.Button("Import")
        import_output = gr.Textbox(label="Kết quả import")
        
        import_button.click(fn=process_file, inputs=file_input, outputs=import_output)
    
    with gr.Tab("Tìm Kiếm"):
        query_input = gr.Textbox(label="Nhập câu hỏi của bạn")
        top_k = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Số lượng kết quả")
        search_button = gr.Button("Tìm kiếm")
        search_output = gr.Markdown(label="Kết quả")
        
        search_button.click(fn=search_documents, inputs=[query_input, top_k], outputs=search_output)

# Khởi chạy ứng dụng
if __name__ == "__main__":
    demo.launch()
