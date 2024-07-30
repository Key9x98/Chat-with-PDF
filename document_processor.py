from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from embedding import custom_embeddings
import os
from typing import List
import re

class DocumentProcessor:
    def __init__(self, pdf_data_path: str, vector_db_path: str):
        self.pdf_data_path = pdf_data_path
        self.vector_db_path = vector_db_path
        self.embedding_model = custom_embeddings

    def process_text(self, text: str) -> str:
        """Xử lý văn bản để loại bỏ khoảng trắng và dòng trống không cần thiết."""
        text = re.sub(r"\n{2,}", "\n\n", text)
        text = re.sub(r"(\S)\n(\S)", r"\1 \2", text)
        return text

    def create_chunks(self, documents: List) -> List:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ".", "!", "?", ""],
            chunk_size=512,
            chunk_overlap=256,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        return [self.process_text(chunk.page_content) for chunk in chunks]

    def create_vector_db(self) -> FAISS:
        loader = DirectoryLoader(self.pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        chunks = self.create_chunks(documents)
        
        db = FAISS.from_texts(chunks, self.embedding_model)
        db.save_local(self.vector_db_path)
        return db

    def run(self):
        return self.create_vector_db()

# Test
if __name__ == "__main__":
    pdf_data_path = 'C:\\Users\\CNTT\\PDFChatbot\\Chat-with-PDF\\PDFs'
    vector_db_path = "vectorstores/db_faiss"
    
    processor = DocumentProcessor(pdf_data_path, vector_db_path)
    db = processor.run()
    print("Vector database created")
