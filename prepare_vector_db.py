import hashlib
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
from embedding import custom_embeddings

# Khai bao bien
pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"
hash_store_path = "vectorstores/hashes.json"

def calculate_file_hash(file_path):
    """Tính toán hash SHA-256 cho một tệp."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Đọc tệp theo từng khối để tránh sử dụng quá nhiều bộ nhớ
        for block in iter(lambda: f.read(4096), b""):
            hasher.update(block)
    return hasher.hexdigest()

def load_existing_hashes():
    """Tải các hash đã lưu từ một tệp."""
    if os.path.exists(hash_store_path):
        with open(hash_store_path, 'r') as f:
            return json.load(f)
    return {}

def save_hashes(hashes):
    """Lưu các hash vào một tệp."""
    with open(hash_store_path, 'w') as f:
        json.dump(hashes, f)

def process_document(document):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1024, 
        chunk_overlap=64,
        length_function=len)
    chunks = text_splitter.split_documents([document])
    return chunks

def create_db_from_files():
    # Khai bao loader de quet toan bo thu muc data
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()

    # Tải các hash đã tồn tại
    existing_hashes = load_existing_hashes()
    new_hashes = existing_hashes.copy()

    # Kiểm tra xem tất cả các tệp PDF có đều đã tồn tại
    all_files_exist = True
    for doc in documents:
        file_path = doc.metadata['source']
        file_hash = calculate_file_hash(file_path)
        if file_hash not in existing_hashes:
            all_files_exist = False
            break

    if all_files_exist:
        print("Tất cả các tệp PDF đã tồn tại trong cơ sở dữ liệu. Không cần cập nhật.")
        return None

    all_chunks = []
    # Sử dụng ThreadPoolExecutor để xử lý đồng thời các tài liệu
    with ThreadPoolExecutor() as executor:
        futures = []
        for doc in documents:
            file_path = doc.metadata['source']
            file_hash = calculate_file_hash(file_path)
            if file_hash not in existing_hashes:
                futures.append(executor.submit(process_document, doc))
                new_hashes[file_hash] = file_path  # Lưu hash mới

        for future in as_completed(futures):
            chunks = future.result()
            all_chunks.extend(chunks)

    # Embedding
    embedding_model = custom_embeddings
    db = FAISS.from_documents(all_chunks, embedding_model)
    db.save_local(vector_db_path)
    
    # Lưu các hash mới
    save_hashes(new_hashes)
    
    return db

# Tạo cơ sở dữ liệu từ các tệp PDF
create_db_from_files()
