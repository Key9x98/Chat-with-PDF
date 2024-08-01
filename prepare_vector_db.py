import hashlib
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
from embedding import custom_embeddings

# Khai bao bien
pdf_data_path = "PDFs"
vector_db_path = "vectorstores/db_faiss"
hash_store_path = "vectorstores/hashes.json"

def calculate_file_hash(file_path):
    """Tính toán hash SHA-256 cho một tệp."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
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
        length_function=len
    )
    chunks = text_splitter.split_documents([document])
    return chunks

def load_existing_db():
    """Tải cơ sở dữ liệu FAISS hiện tại, nếu tồn tại."""
    if os.path.exists(vector_db_path):
        return FAISS.load_local(vector_db_path, custom_embeddings, allow_dangerous_deserialization=True)
    return None

def create_db_from_files():
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    existing_hashes = load_existing_hashes()
    new_hashes = existing_hashes.copy()

    all_chunks = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for doc in documents:
            file_path = doc.metadata['source']
            file_hash = calculate_file_hash(file_path)
            if file_hash not in existing_hashes:
                futures.append(executor.submit(process_document, doc))
                new_hashes[file_hash] = file_path

        for future in as_completed(futures):
            chunks = future.result()
            all_chunks.extend(chunks)

    if all_chunks:
        embedding_model = custom_embeddings
        existing_db = load_existing_db()
        if existing_db:
            existing_db.add_documents(all_chunks)
            existing_db.save_local(vector_db_path)
        else:
            db = FAISS.from_documents(all_chunks, embedding_model)
            db.save_local(vector_db_path)
    else:
        print("Không có tài liệu mới để thêm vào cơ sở dữ liệu.")

    save_hashes(new_hashes)




def is_pdf_exists(file_path):
    file_hash = calculate_file_hash(file_path)
    existing_hashes = load_existing_hashes()
    return file_hash in existing_hashes


def update(file_path):
    file_hash = calculate_file_hash(file_path)
    existing_hashes = load_existing_hashes()

    if file_hash in existing_hashes:
        print(f"Tệp {file_path} đã tồn tại trong cơ sở dữ liệu.")
        return

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    all_chunks = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_document, doc) for doc in documents]
        for future in as_completed(futures):
            chunks = future.result()
            all_chunks.extend(chunks)

    if all_chunks:
        embedding_model = custom_embeddings
        existing_db = load_existing_db()
        if existing_db:
            existing_db.add_documents(all_chunks)
            existing_db.save_local(vector_db_path)
        else:
            db = FAISS.from_documents(all_chunks, embedding_model)
            db.save_local(vector_db_path)
    else:
        print("Không có tài liệu mới để thêm vào cơ sở dữ liệu.")

    existing_hashes[file_hash] = file_path
    save_hashes(existing_hashes)



import os

def run(pdf_directory):
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    for pdf_file in pdf_files:
        file_path = os.path.join(pdf_directory, pdf_file)
        print(f"Đang kiểm tra tệp: {file_path}")
        if not is_pdf_exists(file_path):
            print(f"Tệp {pdf_file} chưa có trong cơ sở dữ liệu. Đang cập nhật...")
            update(file_path)
        else:
            print(f"Tệp {pdf_file} đã có trong cơ sở dữ liệu.")

# Ví dụ sử dụng hàm run
pdf_data_path = "PDFs"
run(pdf_data_path)

user_question = "tóm tắt trích xuất là gì"
db = load_existing_db()
a = db.similarity_search(user_question, k=2)
print(a)
