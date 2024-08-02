import hashlib
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
from embedding import custom_embeddings

class PDFDatabaseManager:
    def __init__(self, pdf_data_path, vector_db_path, hash_store_path):
        self.pdf_data_path = pdf_data_path
        self.vector_db_path = vector_db_path
        self.hash_store_path = hash_store_path

    def calculate_file_hash(self, file_path):
        """Tính toán hash SHA-256 cho một tệp."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(4096), b""):
                hasher.update(block)
        return hasher.hexdigest()

    def load_existing_hashes(self):
        """Tải các hash đã lưu từ một tệp."""
        if os.path.exists(self.hash_store_path):
            with open(self.hash_store_path, 'r') as f:
                return json.load(f)
        return {}

    def save_hashes(self, hashes):
        """Lưu các hash vào một tệp."""
        with open(self.hash_store_path, 'w') as f:
            json.dump(hashes, f)

    def process_document(self, document):
        """Xử lý tài liệu để chia thành các đoạn văn bản."""
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ".", "!", "?", ""],
            chunk_size=512,
            chunk_overlap=256,
            length_function=len
        )
        chunks = text_splitter.split_documents([document])
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i

        return chunks

    def load_existing_db(self):
        """Tải cơ sở dữ liệu FAISS hiện tại, nếu tồn tại."""
        if os.path.exists(self.vector_db_path):
            return FAISS.load_local(self.vector_db_path, custom_embeddings, allow_dangerous_deserialization=True)
        return None

    def create_db_from_files(self):
        """Tạo hoặc cập nhật cơ sở dữ liệu từ các tệp PDF trong thư mục."""
        loader = DirectoryLoader(self.pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        existing_hashes = self.load_existing_hashes()
        new_hashes = existing_hashes.copy()

        all_chunks = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for doc in documents:
                file_path = doc.metadata['source']
                file_hash = self.calculate_file_hash(file_path)
                if file_hash not in existing_hashes:
                    futures.append(executor.submit(self.process_document, doc))
                    new_hashes[file_hash] = file_path

            for future in as_completed(futures):
                chunks = future.result()
                all_chunks.extend(chunks)

        if all_chunks:
            embedding_model = custom_embeddings
            existing_db = self.load_existing_db()
            if existing_db:
                existing_db.add_documents(all_chunks)
                existing_db.save_local(self.vector_db_path)
            else:
                db = FAISS.from_documents(all_chunks, embedding_model)
                db.save_local(self.vector_db_path)
        else:
            print("Không có tài liệu mới để thêm vào cơ sở dữ liệu.")

        self.save_hashes(new_hashes)

    def is_pdf_exists(self, file_path):
        """Kiểm tra xem tệp PDF có tồn tại trong cơ sở dữ liệu hay không."""
        file_hash = self.calculate_file_hash(file_path)
        existing_hashes = self.load_existing_hashes()
        return file_hash in existing_hashes

    def update_db(self, file_path):
        file_hash = self.calculate_file_hash(file_path)
        existing_hashes = self.load_existing_hashes()

        if file_hash in existing_hashes:
            print(f"Tệp {file_path} đã tồn tại trong cơ sở dữ liệu.")
            return

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        all_chunks = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_document, doc) for doc in documents]
            for future in as_completed(futures):
                chunks = future.result()
                all_chunks.extend(chunks)

        if all_chunks:
            embedding_model = custom_embeddings
            existing_db = self.load_existing_db()
            if existing_db:
                existing_db.add_documents(all_chunks)
                existing_db.save_local(self.vector_db_path)
            else:
                db = FAISS.from_documents(all_chunks, embedding_model)
                db.save_local(self.vector_db_path)
        else:
            print("Không có tài liệu mới để thêm vào cơ sở dữ liệu.")

        existing_hashes[file_hash] = file_path
        self.save_hashes(existing_hashes)


## test
pdf_data_path = "PDFs"
vector_db_path = "vectorstores/db_faiss"
hash_store_path = "vectorstores/hashes.json"

manager = PDFDatabaseManager(pdf_data_path, vector_db_path, hash_store_path)

# pdf_files = [f for f in os.listdir(pdf_data_path) if f.lower().endswith('.pdf')]
# for pdf_file in pdf_files:
#     file_path = os.path.join(pdf_data_path, pdf_file)
#     print(f"Đang kiểm tra tệp: {file_path}")
#     if not manager.is_pdf_exists(file_path):
#         print(f"Tệp {pdf_file} chưa có trong db. Processing...")
#         manager.update_db(file_path)
#     else:
#         print(f"Tệp {pdf_file} đã có trong db, gửi tệp khác.")

#
user_question = ("Bộ dữ liệu")
db = manager.load_existing_db()
a = db.similarity_search(user_question, k=3)

context = "\n-----------------------------------\n".join([
    f"Content:\n{doc.page_content}\nMetadata:\n{doc.metadata}" for doc in a
])
print(context)

# original_docs = manager.process_document()
# print(original_docs)


# loader = PyPDFLoader('C:\\Users\\CNTT\\PDFChatbot\\Chat-with-PDF\\PDFs\\Hoàng_Vũ_Minh_KLTN_HUS.pdf')
# documents = loader.load()
#
# all_text = "\n".join(doc.page_content for doc in documents)
# print(all_text)
# with open('text.txt', 'w', encoding='utf-8') as file:
#     file.write(all_text)
# Xử lý văn bản

