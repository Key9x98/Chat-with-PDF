from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from embedding import custom_embeddings
from typing import List
import re

# Khai bao bien
pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

# def text_splitter(text: str) -> List[str]:
#     chunks = []
#     paragraphs = text.split('\n')
#     current_chunk = []
#     current_word_count = 0

#     for paragraph in paragraphs:
#         words = paragraph.split()
#         if current_word_count + len(words) >= 100:
#             if current_chunk:
#                 chunks.append(' '.join(current_chunk))
#             current_chunk = words
#             current_word_count = len(words)
#         else:
#             current_chunk.extend(words)
#             current_word_count += len(words)
#     if current_chunk:
#         chunks.append(' '.join(current_chunk))
    
#     return chunks

def extract_text(docs):
    docs = re.sub(r"\n{2,}", "\n\n", docs)
    docs = re.sub(
        r"(\S)\n(\S)", r"\1 \2", docs
    )
    return docs
# Hàm để xử lý page_content trong từng chunk
def process_chunks(chunks):
    for chunk in chunks:
        chunk.page_content = extract_text(chunk.page_content)
    return chunks
#seperator tốt: "\n\n", "\n", " ", ".", "!", "?",""
#chunk_size = 1024, chunk_overlap = 64
def create_db_from_files():
    # Khai bao loader de quet toan bo thu muc dataa
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ", 
            ".", 
            "!", 
            "?",
            ""
        ],
        chunk_size=1024,
        chunk_overlap=64
    )
    chunks = text_splitter.split_documents(documents)
    processed_chunks = process_chunks(chunks)
    # print(processed_chunks)
    # Create document objects from chunks
    # documents = [Document(page_content=chunk.page_content) for chunk in chunks]
    # Embedding
    embedding_model = custom_embeddings
    db = FAISS.from_documents(processed_chunks, embedding_model)
    db.save_local(vector_db_path)
    return db

# Tạo cơ sở dữ liệu từ các tệp PDF
create_db_from_files()
