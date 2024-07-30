from langchain_community.vectorstores import FAISS

class Retriever:
    def __init__(self, vector_db_path):
        self.vector_db_path = vector_db_path
        self.embedding_model = custom_embeddings
        self.db = None
        self.retriever = None

    def load_vector_db(self):
        """Load vector database from local storage."""
        self.db = FAISS.load_local(
            self.vector_db_path, 
            self.embedding_model, 
            allow_dangerous_deserialization=True
        )
        self.retriever = self.db.as_retriever()

    def get_relevant_documents(self, question):
        """Retrieve relevant documents based on the question."""
        if not self.retriever:
            self.load_vector_db()
        return self.retriever.invoke(question)

    def search_similar_documents(self, query, k=5):
        """Search for similar documents using the vector database."""
        if not self.db:
            self.load_vector_db()
        return self.db.similarity_search(query, k=k)

    def get_retriever(self):
        """Get the retriever object."""
        if not self.retriever:
            self.load_vector_db()
        return self.retriever


vector_db_path = "vectorstores/db_faiss"
retriever = Retriever(vector_db_path)

question = "Điều kiện nghỉ học là gì"
relevant_docs = retriever.get_relevant_documents(question)
print(f"Found {len(relevant_docs)} relevant documents")

similar_docs = retriever.search_similar_documents(question, k=3)
print(f"Found {len(similar_docs)} similar documents")
