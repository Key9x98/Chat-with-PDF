from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from embedding import custom_embeddings
# Khai bao bien
pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

def create_db_from_files():
    # Khai bao loader de quet toan bo thu muc dataa
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=512, 
        chunk_overlap=128,
        length_function=len)
    chunks = text_splitter.split_documents(documents)   
    # Create document objects from chunks
    # documents = [Document(page_content=chunk.page_content) for chunk in chunks]
    # Embedding
    embedding_model = custom_embeddings
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db

# Tạo cơ sở dữ liệu từ các tệp PDF
create_db_from_files()
