import os
import tempfile
import time
import uuid

import gradio as gr
import weaviate
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Load biến môi trường
load_dotenv()

# Lấy API keys và cài đặt
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")

# Kết nối Weaviate
def connect_to_weaviate():
    try:
        client = weaviate.Client(
            url="http://localhost:8080",
        )
        print(f"Weaviate connection status: {client.is_ready()}")
        return client
    except Exception as e:
        print(f"Không thể kết nối với Weaviate: {e}")
        return None

client = connect_to_weaviate()

# Tên collection
COLLECTION_NAME = "RAGDocuments"

# Tạo collection nếu chưa tồn tại
def create_schema():
    if client is None:
        return "Không thể kết nối với Weaviate"
    
    schema = {
        "classes": [
            {
                "class": COLLECTION_NAME,
                "description": "Document chunks for RAG",
                "vectorizer": "text2vec-transformers",
                "properties": [
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
                        "dataType": ["text"],
                        "description": "The chunk ID within the document",
                    }
                ],
            }
        ]
    }
    
    try:
        # Kiểm tra xem schema đã tồn tại chưa
        current_schema = client.schema.get()
        existing_classes = [c["class"] for c in current_schema["classes"]]
        
        if COLLECTION_NAME not in existing_classes:
            client.schema.create(schema)
            print(f"Schema '{COLLECTION_NAME}' đã được tạo thành công")
        else:
            print(f"Schema '{COLLECTION_NAME}' đã tồn tại")
        
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
def get_llm(provider=DEFAULT_LLM_PROVIDER, temperature=0):
    if provider == "openai" and OPENAI_API_KEY:
        return ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",
            temperature=temperature
        )
    elif provider == "gemini" and GOOGLE_API_KEY:
        return ChatGoogleGenerativeAI(
            google_api_key=GOOGLE_API_KEY,
            model="gemini-pro",
            temperature=temperature
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
        # Lưu file tạm
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        file_obj.save(temp_file.name)
        temp_path = temp_file.name
        
        # Lấy tên file gốc và phần mở rộng
        original_filename = file_obj.name
        file_extension = os.path.splitext(original_filename)[1].lower()
        
        # Xử lý dựa trên loại file
        if file_extension == '.pdf':
            loader = PyPDFLoader(temp_path)
        elif file_extension == '.txt':
            loader = TextLoader(temp_path)
        else:
            os.unlink(temp_path)
            return f"Không hỗ trợ định dạng file {file_extension}"
        
        # Load và chia nhỏ văn bản
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        # Import các chunk vào Weaviate
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            client.data_object.create(
                {
                    "content": chunk.page_content,
                    "source": original_filename,
                    "chunk_id": chunk_id
                },
                COLLECTION_NAME
            )
        
        # Xóa file tạm
        os.unlink(temp_path)
        
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
        result = (
            client.query
            .get(COLLECTION_NAME, ["content", "source"])
            .with_near_text({"concepts": [query]})
            .with_limit(top_k)
            .do()
        )
        
        objects = result["data"]["Get"][COLLECTION_NAME]
        
        if not objects:
            return "Không tìm thấy kết quả phù hợp"
        
        formatted_results = []
        for i, obj in enumerate(objects):
            formatted_results.append(f"### Kết quả {i+1}\n")
            formatted_results.append(f"**Nguồn:** {obj['source']}\n")
            formatted_results.append(f"**Nội dung:**\n{obj['content']}\n\n")
        
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
        provider = gr.Radio(
            ["gemini", "openai"], 
            label="Chọn LLM", 
            value=DEFAULT_LLM_PROVIDER
        )
        rag_query = gr.Textbox(label="Nhập câu hỏi của bạn")
        rag_button = gr.Button("Trả lời")
        rag_output = gr.Markdown(label="Câu trả lời")
        
        rag_button.click(fn=answer_question, inputs=[rag_query, provider], outputs=rag_output)

# Khởi chạy ứng dụng
if __name__ == "__main__":
    demo.launch() 