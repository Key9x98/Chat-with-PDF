import hashlib
import os
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
from embedding import custom_embeddings
from text_processor import TextProcessor

text_processor = TextProcessor()


class PDFDatabaseManager:
    def __init__(self, pdf_data_path, vector_db_path, hash_store_path):
        self.pdf_data_path = pdf_data_path
        self.vector_db_path = vector_db_path
        self.hash_store_path = hash_store_path
        self.pdf_databases = {}

    def calculate_file_hash(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(4096), b""):
                hasher.update(block)
        return hasher.hexdigest()

    def load_existing_hashes(self):
        try:
            if os.path.exists(self.hash_store_path):
                with open(self.hash_store_path, 'r') as f:
                    return json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {self.hash_store_path}. Starting with empty hash store.")
        return {}

    def save_hashes(self, hashes):
        try:
            with open(self.hash_store_path, 'w') as f:
                json.dump(hashes, f)
        except IOError as e:
            print(f"Error saving hashes: {e}")

    def process_document(self, document):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ".", "!", "?", ""],
            chunk_size=512,
            chunk_overlap=256,
            length_function=len
        )
        chunks = text_splitter.split_documents([document])
        return chunks

    def load_existing_db(self, file_name):
        db_path = os.path.join(self.vector_db_path, file_name)
        if os.path.exists(db_path):
            try:
                return FAISS.load_local(db_path, custom_embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"Error loading database for {file_name}: {e}")
        return None

    def is_pdf_exists(self, file_path):
        if not os.path.exists(file_path):
            return False
        file_hash = self.calculate_file_hash(file_path)
        existing_hashes = self.load_existing_hashes()
        return file_hash in existing_hashes

    def update_db(self, file_path):
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None

        file_hash = self.calculate_file_hash(file_path)
        existing_hashes = self.load_existing_hashes()

        if file_hash in existing_hashes:
            print(f"File {file_path} already exists in the database.")
            return None

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        output_dir = 'original_text'
        os.makedirs(output_dir, exist_ok=True)
        file_name = os.path.basename(file_path)

        # bỏ dấu đi, vì faiss không nhận có dấu tviet
        file_name_without_ext = text_processor.remove_accents(os.path.splitext(file_name)[0])
        output_file_name = f"{file_name_without_ext}.txt"

        output_file_path = os.path.join(output_dir, output_file_name)

        all_text = "\n".join(doc.page_content for doc in documents)
        try:
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(all_text)
        except IOError as e:
            print(f"Error writing to file {output_file_path}: {e}")
            return None

        chunks = []
        for doc in documents:
            page_chunks = self.process_document(doc)
            chunks.extend(page_chunks)

        if chunks:
            embedding_model = custom_embeddings
            db = FAISS.from_documents(chunks, embedding_model)
            db_path = os.path.join(self.vector_db_path, file_name_without_ext)
            db.save_local(db_path)
            existing_hashes[file_hash] = file_path
            self.save_hashes(existing_hashes)
            return db
        else:
            print("No new documents to add to the database.")
            return None


class ContextRetriever:
    def __init__(self, context_dir='original_text'):
        self.context_dir = context_dir

    def read_text_file(self, file_name):
        file_path = os.path.join(self.context_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                all_text = file.read()
            return all_text
        except IOError as e:
            print(f"Error reading file {file_path}: {e}")
            return ""

    def expand_context(self, file_name, context, num_words_before=200, num_words_after=200):
        all_text = self.read_text_file(file_name)
        if not all_text:
            return "Context not found"

        match = re.search(re.escape(context), all_text)
        if not match:
            return "Context not found in the document"

        start, end = match.span()

        before_context = all_text[:start].split()
        before_context = before_context[max(0, len(before_context) - num_words_before):]

        after_context = all_text[end:].split()
        after_context = after_context[:min(len(after_context), num_words_after)]

        expanded_context = " ".join(before_context) + " " + context + " " + " ".join(after_context)

        return expanded_context

    def get_file_name(self, metadata):
        source = metadata.get('source', '')
        file_name = os.path.basename(source)
        file_name, _ = os.path.splitext(file_name)
        file_name_txt = file_name + '.txt'
        return file_name_txt