from sentence_transformers import SentenceTransformer
import torch

class DocumentEmbedder:
    def __init__(self, model_name="hiieu/halong_embedding"):
        self.model = SentenceTransformer(model_name)
    
    def convert_to_vec(self, docs):
        """
        encode single docs / list of docs to vectors
        """
        return self.model.encode(docs)

halong = DocumentEmbedder()
text = "test"
embed=  halong.convert_to_vec(text)
print(embed)
