from sentence_transformers import SentenceTransformer
import numpy as np
from langchain.embeddings.base import Embeddings

model_name = 'hiieu/halong_embedding'  # Thay đổi nếu cần sử dụng mô hình khác
model = SentenceTransformer(model_name)

texts = [
    "Deep Learning là một nhánh của học máy, sử dụng các mạng nơ-ron sâu để mô hình hóa và giải quyết các vấn đề phức tạp. "
    "Nó có nhiều ứng dụng trong thực tế như nhận diện hình ảnh, xử lý ngôn ngữ tự nhiên, và chẩn đoán y tế.",
    
    "Món ăn hôm nay rất ngon",
    "Mạng Neural học sâu học rất sâu nên được gọi là Deep Learning"
]

class CustomEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts):
        return self.model.encode(texts)
    
    def embed_query(self, text):
        return self.model.encode([text])[0]

custom_embeddings = CustomEmbeddings(model)
