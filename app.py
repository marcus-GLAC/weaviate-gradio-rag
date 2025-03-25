import os
import shutil
import tempfile
import time
import uuid

import gradio as gr
import weaviate
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Weaviate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Load biến môi trường
load_dotenv()

# Lấy API keys và cài đặt
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")

# Lấy cài đặt mô hình
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0"))

# Lấy URL Weaviate từ biến môi trường hoặc dùng giá trị mặc định
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
UPLOADS_DIR = os.getenv("UPLOADS_DIR", "./uploads")

# Đảm bảo thư mục uploads tồn tại
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Kết nối Weaviate
def connect_to_weaviate():
    try:
        # Lấy URL từ biến môi trường
        weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        print(f"Đang kết nối đến Weaviate tại: {weaviate_url}")
        
        # Thử ping trước khi kết nối
        import requests
        try:
            response = requests.get(f"{weaviate_url}/v1/meta")
            if response.status_code != 200:
                print(f"Weaviate không phản hồi tại {weaviate_url}, status code: {response.status_code}")
                print(f"Response: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Không thể ping đến Weaviate: {e}")
        
        # Kết nối sử dụng Weaviate Client v4 thay vì v3
        client = weaviate.WeaviateClient(
            connection_params=weaviate.connect.ConnectionParams.from_url(
                url=weaviate_url,
                timeout_config=(5, 15)  # (Kết nối timeout, Đọc timeout)
            )
        )
        
        # Kiểm tra kết nối 
        is_ready = client.is_ready()
        print(f"Weaviate connection status: {is_ready}")
        
        if not is_ready:
            print("Weaviate kết nối được nhưng báo chưa sẵn sàng")
            
        return client
    except Exception as e:
        print(f"Lỗi kết nối đến Weaviate: {e}")
        print("Vui lòng kiểm tra:")
        print("1. Docker và container Weaviate đã chạy chưa? (docker ps)")
        print("2. URL kết nối có đúng không? (mặc định: http://localhost:8080)")
        print("3. Weaviate container có lỗi không? (docker logs weaviate-db)")
        return None

client = connect_to_weaviate()

# Tên collection
COLLECTION_NAME = "RAGDocuments"

# Tạo collection nếu chưa tồn tại
def create_schema():
    if client is None:
        return "Không thể kết nối với Weaviate"
    
    try:
        # Kiểm tra xem collection đã tồn tại chưa
        try:
            # Thử lấy collection
            client.collections.get(COLLECTION_NAME)
            print(f"Collection '{COLLECTION_NAME}' đã tồn tại")
            return True
        except:
            # Collection chưa tồn tại, tạo mới
            client.collections.create(
                name=COLLECTION_NAME,
                description="Document chunks for RAG",
                properties=[
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "The content of the document chunk",
                        "indexSearchable": True,
                    },
                    {
                        "name": "source",
                        "dataType": ["text"],
                        "description": "The source file of the document",
                        "indexSearchable": True,
                    },
                    {
                        "name": "chunk_id",
                        "dataType": ["text"],
                        "description": "The chunk ID within the document",
                    }
                ],
                vectorizer_config=weaviate.classes.Configure.Vectorizer.text2vec_transformers(),
            )
            print(f"Collection '{COLLECTION_NAME}' đã được tạo thành công")
            return True
    
    except Exception as e:
        print(f"Lỗi khi tạo schema: {e}")
        return False

# Tạo schema
create_schema()

# Thiết lập embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)

# Tạo Weaviate vector store cho LangChain
def get_vectorstore():
    if client is None:
        return None
    
    return Weaviate(
        client=client,
        index_name=COLLECTION_NAME,
        text_key="content",
        embedding=embeddings,
    )

# Tạo retriever
def get_retriever(k=5):
    vectorstore = get_vectorstore()
    if vectorstore:
        return vectorstore.as_retriever(search_kwargs={"k": k})
    return None

# Khởi tạo LLM dựa trên provider
def get_llm(provider=DEFAULT_LLM_PROVIDER, temperature=None):
    if provider == "openai" and OPENAI_API_KEY:
        # Đảm bảo model là một trong những model được cho phép
        model_name = OPENAI_MODEL
        if model_name not in ["gpt-4o", "gpt-4o-mini"]:
            model_name = "gpt-4o-mini"  # Mặc định nếu cấu hình không hợp lệ
            
        return ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name=model_name,
            temperature=OPENAI_TEMPERATURE if temperature is None else temperature
        )
    elif provider == "gemini" and GOOGLE_API_KEY:
        return ChatGoogleGenerativeAI(
            google_api_key=GOOGLE_API_KEY,
            model="gemini-2.0-flash",  # Chỉ cho phép model này
            temperature=GEMINI_TEMPERATURE if temperature is None else temperature
        )
    else:
        return None

# Tạo QA Chain
def get_qa_chain(provider=DEFAULT_LLM_PROVIDER):
    llm = get_llm(provider)
    retriever = get_retriever()
    
    if llm and retriever:
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
    return None

# Hàm xử lý và import file
def process_file(file_obj):
    if client is None:
        return "Lỗi: Không thể kết nối với Weaviate"
    
    if file_obj is None:
        return "Vui lòng chọn file để import"
    
    try:
        # Lấy tên file gốc và phần mở rộng
        original_filename = file_obj.name
        file_extension = os.path.splitext(original_filename)[1].lower()
        
        # Tạo đường dẫn đến file trong thư mục uploads
        timestamp = int(time.time())
        saved_filename = f"{timestamp}_{original_filename}"
        saved_path = os.path.join(UPLOADS_DIR, saved_filename)
        
        # Lưu file
        with open(saved_path, 'wb') as f:
            shutil.copyfileobj(file_obj.file, f)
        
        # Xử lý dựa trên loại file
        if file_extension == '.pdf':
            loader = PyPDFLoader(saved_path)
        elif file_extension == '.txt':
            loader = TextLoader(saved_path)
        else:
            os.remove(saved_path)
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
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            collection.data.insert(
                properties={
                    "content": chunk.page_content,
                    "source": original_filename,
                    "chunk_id": chunk_id
                }
            )
        
        return f"Đã import thành công {len(chunks)} chunks từ file {original_filename}"
    except Exception as e:
        return f"Lỗi khi xử lý file: {str(e)}"

# Hàm tìm kiếm
def search_documents(query, top_k=5):
    if client is None:
        return "Lỗi: Không thể kết nối với Weaviate"
    
    if not query:
        return "Vui lòng nhập câu hỏi để tìm kiếm"
    
    try:
        # Lấy collection
        collection = client.collections.get(COLLECTION_NAME)
        
        # Tìm kiếm semantically
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
    except Exception as e:
        return f"Lỗi khi tìm kiếm: {str(e)}"

# Hàm trả lời câu hỏi với RAG
def answer_question(query, provider=DEFAULT_LLM_PROVIDER):
    if not query:
        return "Vui lòng nhập câu hỏi"
    
    qa_chain = get_qa_chain(provider)
    
    if not qa_chain:
        if provider == "openai":
            return "LLM không khả dụng. Vui lòng kiểm tra OPENAI_API_KEY trong file .env"
        else:
            return "LLM không khả dụng. Vui lòng kiểm tra GOOGLE_API_KEY trong file .env"
    
    try:
        result = qa_chain({"query": query})
        
        answer = result["result"]
        sources = [doc.metadata["source"] for doc in result["source_documents"] if "source" in doc.metadata]
        unique_sources = list(set(sources))
        
        formatted_answer = f"### Câu trả lời:\n{answer}\n\n### Nguồn tham khảo:\n"
        for src in unique_sources:
            formatted_answer += f"- {src}\n"
        
        return formatted_answer
    except Exception as e:
        return f"Lỗi khi xử lý câu hỏi: {str(e)}"

# Tạo giao diện Gradio
with gr.Blocks(title="RAG với Weaviate và Gemini/OpenAI") as demo:
    gr.Markdown("# Ứng dụng RAG với Weaviate, Gradio và Gemini/OpenAI")
    
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
    
    with gr.Tab("Chat RAG"):
        with gr.Row():
            with gr.Column(scale=1):
                provider = gr.Radio(
                    ["gemini", "openai"], 
                    label="Chọn LLM Provider", 
                    value=DEFAULT_LLM_PROVIDER,
                    interactive=True
                )
            with gr.Column(scale=2):
                model_dropdown = gr.Dropdown(
                    label="Chọn Mô Hình",
                    choices={
                        "gemini": ["gemini-2.0-flash"],
                        "openai": ["gpt-4o-mini", "gpt-4o"]
                    }[DEFAULT_LLM_PROVIDER],
                    value="gemini-2.0-flash" if DEFAULT_LLM_PROVIDER == "gemini" else "gpt-4o-mini",
                    interactive=True
                )
        
        temperature = gr.Slider(
            minimum=0, 
            maximum=1, 
            value=0.1,
            step=0.1, 
            label="Temperature (độ sáng tạo)",
            info="Giá trị thấp cho câu trả lời chính xác, giá trị cao cho câu trả lời sáng tạo"
        )
        
        # Cập nhật lựa chọn mô hình khi thay đổi provider
        def update_model_choices(provider_value):
            if provider_value == "gemini":
                return {
                    "choices": ["gemini-2.0-flash"],
                    "value": "gemini-2.0-flash",
                }
            else:
                return {
                    "choices": ["gpt-4o-mini", "gpt-4o"],
                    "value": "gpt-4o-mini",
                }
        
        provider.change(
            fn=update_model_choices,
            inputs=provider,
            outputs=model_dropdown,
        )
        
        with gr.Row():
            with gr.Column(scale=4):
                rag_query = gr.Textbox(
                    label="Nhập câu hỏi của bạn",
                    placeholder="Nhập câu hỏi để tìm thông tin từ tài liệu...",
                    lines=2
                )
            with gr.Column(scale=1):
                rag_button = gr.Button("Trả lời", variant="primary")
        
        rag_output = gr.Markdown(label="Câu trả lời")
        
        # Các nguồn tham khảo
        sources_output = gr.Markdown(label="Nguồn tham khảo", visible=False)
        
        # Cập nhật hàm answer_question để sử dụng model và temperature
        def answer_with_model(query, provider, model, temp):
            if not query:
                return "Vui lòng nhập câu hỏi", ""
            
            # Giới hạn model theo provider
            if provider == "gemini":
                actual_model = "gemini-2.0-flash"  # Chỉ cho phép model này
            else:
                if model not in ["gpt-4o-mini", "gpt-4o"]:
                    actual_model = "gpt-4o-mini"  # Mặc định
                else:
                    actual_model = model
            
            # Cập nhật biến môi trường tạm thời
            if provider == "gemini":
                os.environ["GEMINI_MODEL"] = actual_model
                os.environ["GEMINI_TEMPERATURE"] = str(temp)
            else:
                os.environ["OPENAI_MODEL"] = actual_model
                os.environ["OPENAI_TEMPERATURE"] = str(temp)
            
            # Gọi hàm LLM với mô hình và temperature đã chọn
            answer = answer_question(query, provider)
            
            # Tách câu trả lời và nguồn tham khảo
            if "### Nguồn tham khảo:" in answer:
                parts = answer.split("### Nguồn tham khảo:")
                answer_text = parts[0].strip()
                sources = "### Nguồn tham khảo:" + parts[1]
                return answer_text, sources
            
            return answer, ""
        
        # Cập nhật hàm xử lý khi nhấn nút
        def on_rag_button_click(query, provider, model, temp):
            answer, sources = answer_with_model(query, provider, model, temp)
            return answer, sources, gr.update(visible=bool(sources))
        
        rag_button.click(
            fn=on_rag_button_click, 
            inputs=[rag_query, provider, model_dropdown, temperature], 
            outputs=[rag_output, sources_output, sources_output]
        )

# Khởi chạy ứng dụng
if __name__ == "__main__":
    # Lấy cấu hình từ biến môi trường
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", 7860))
    root_path = os.getenv("GRADIO_ROOT_PATH", "/")
    
    # In thông tin truy cập
    print(f"Khởi động ứng dụng Gradio trên {server_name}:{server_port}")
    if server_name == "0.0.0.0":
        print(f"Bạn có thể truy cập ứng dụng qua địa chỉ IP của máy chủ: http://IP-ADDRESS:{server_port}")
    
    # Khởi động Gradio
    demo.queue().launch(
        server_name=server_name,
        server_port=server_port,
        root_path=root_path
    ) 