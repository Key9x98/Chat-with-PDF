from sentence_transformers import SentenceTransformer
import torch
from langchain.embeddings import Embeddings


class DocumentEmbedder(Embeddings):
    def __init__(self, model_name="hiieu/halong_embedding"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, documents):
        return self.model.encode(documents)
    
    def embed_query(self, query):
        return self.model.encode([query])[0]
