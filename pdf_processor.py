import hashlib
import os
import json
import re
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
        if os.path.exists(self.hash_store_path):
            with open(self.hash_store_path, 'r') as f:
                return json.load(f)
        return {}

    def save_hashes(self, hashes):
        with open(self.hash_store_path, 'w') as f:
            json.dump(hashes, f)

    def process_document(self, document):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ".", "!", "?", ""],
            chunk_size=512,
            chunk_overlap=256,
            length_function=len
        )
        chunks = text_splitter.split_documents([document])

        return chunks

    def load_existing_db(self):
        if os.path.exists(self.vector_db_path):
            return FAISS.load_local(self.vector_db_path, custom_embeddings, allow_dangerous_deserialization=True)
        return None

    def create_db_from_files(self):
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
        '''
        Dùng hash check xem file pdf đã ton tai chua

        '''
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

        output_dir = 'original_text'
        os.makedirs(output_dir, exist_ok=True)
        file_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(file_name)[0]
        output_file_name = f"{file_name_without_ext}.txt"

        # Tạo đường dẫn đầy đủ tới file đầu ra
        output_file_path = os.path.join(output_dir, output_file_name)

        # Lưu nội dung của file PDF vào file .txt
        all_text = "\n".join(doc.page_content for doc in documents)
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(all_text)

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


class ContextRetriever:
    def __init__(self, context_dir='original_text'):
        self.context_dir = context_dir

    def read_text_file(self, file_name):
        '''
        input: file_name txt chứa all text
        return: string all text
        '''
        file_path = os.path.join(self.context_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            # all_text = file.read().replace('\n', ' ')
            all_text = file.read()
        return all_text

    def expand_context(self, file_name, context, num_words_before=200, num_words_after=200):
        '''
        Tìm context trong văn bản chính sau đó mở rộng ra trước và sau số từ chỉ định
        '''

        # context = context.replace('\n', ' ')
        all_text = self.read_text_file(file_name)
        match = re.search(re.escape(context), all_text)
        if not match:
            return "Không tìm thấy context"
        start, end = match.span()

        before_context = all_text[:start].split()
        before_context = before_context[max(0, len(before_context) - num_words_before):]

        after_context = all_text[end:].split()
        after_context = after_context[:min(len(after_context), num_words_after)]

        expanded_context = " ".join(before_context) + " " + context + " " + " ".join(after_context)

        return expanded_context

    def get_file_name(self, metadata):
        '''
        Trích xuất tên file từ metadata
        Sau đấy chuyển đuôi pdf thành đuôi txt để chuyền vào hàm read_text_file
        '''
        # Trích xuất giá trị của trường 'source' từ metadata
        source = metadata.get('source', '')
        # Tìm tên file trong đường dẫn của trường 'source'
        file_name = os.path.basename(source)
        file_name, _ = os.path.splitext(file_name)
        file_name_txt = file_name + '.txt'
        return file_name_txt

# TEST EMBEDDING
# pdf_data_path = "PDFs"
# vector_db_path = "vectorstores/db_faiss"
# hash_store_path = "vectorstores/hashes.json"
#
# manager = PDFDatabaseManager(pdf_data_path, vector_db_path, hash_store_path)

# import time
# start_embed_pt = time.time()
# pdf_files = [f for f in os.listdir(pdf_data_path) if f.lower().endswith('.pdf')]
# for pdf_file in pdf_files:
#     file_path = os.path.join(pdf_data_path, pdf_file)
#     print(f"Đang kiểm tra tệp: {file_path}")
#     if not manager.is_pdf_exists(file_path):
#         print(f"Tệp {pdf_file} chưa có trong db. Processing...")
#         manager.update_db(file_path)
#     else:
#         print(f"Tệp {pdf_file} đã có trong db, gửi tệp khác.")
# end_embed_pt = time.time()
# print("Pytorch embed time:", end_embed_pt  - start_embed_pt)

# TEST FINDING AND EXPANDING CONTEXT
# print("=========================================")
# user_question = ("Sinh viên cần đạt những điều kiện gì để được xét học bổng  khuyến khích tập")
# db = manager.load_existing_db()
# a = db.similarity_search(user_question, k=2)
#
# context = [doc.page_content for doc in a]
# metadata = [doc.metadata for doc in a]
#
# retriever = ContextRetriever()
#
# expanded_contexts = []
#
# for i in range(len(context)):
#     # Lấy tên file tương ứng với metadata
#     file_name = retriever.get_file_name(metadata[i])
#     print(f"Tên file là: {file_name}")
#
#     expanded_context = retriever.expand_context(file_name, context[i])
#     expanded_contexts.append(expanded_context)
#
# final_context = "\n--------------------------\n".join(expanded_contexts)
#
# # Ghi final_context vào một file văn bản
# with open('text.txt', 'w', encoding='utf-8') as file:
#     file.write(final_context)
#
#
# print("Combined Expanded Context:\n", final_context)







