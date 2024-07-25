import google.generativeai as genai
import os
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Đường dẫn tới cơ sở dữ liệu vector
vector_db_path = "vectorstores/db_faiss"

# Tải .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# Thiết lập cấu hình sinh
generation_config = {
    "temperature": 0.05,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

# Định nghĩa lớp GeminiBot
class GeminiBot:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.token_count = None
        self.model = None
        self.chat = None
        self._setup()

    def _setup(self):
        genai.configure(api_key=GOOGLE_API_KEY)
        PROMPT = 'Bạn là một công cụ chat, hãy trả lời các câu hỏi của người dùng'
        self.model = genai.GenerativeModel(model_name=self.model_name,
                                           generation_config=generation_config,
                                           safety_settings=safety_settings)
                                           # system_instruction=PROMPT)
        self.token_count = self.model.count_tokens(PROMPT).total_tokens
        self.chat = self.model.start_chat()

    def response(self, user_input):
        user_input = user_input
        input_tokens = self.model.count_tokens(user_input).total_tokens
        if self.token_count > 1000000:
            return "Out of tokens"
        prompt = "Bạn là một hệ thống hỏi đáp, hãy giúp người dùng trả lời câu hỏi theo cú pháp câu hỏi - câu trả lời"
        response = self.chat.send_message(prompt + user_input)
        self.token_count += input_tokens + self.model.count_tokens(response.text).total_tokens
        return response.text

# Đọc cơ sở dữ liệu vector
def read_vectors_db():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

# Tạo hàm để trả lời câu hỏi dựa vào cơ sở dữ liệu vector
def answer_question(question):
    # Đọc cơ sở dữ liệu vector
    db = read_vectors_db()
    retriever = db.as_retriever()
    # embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    query = question
    documents = retriever.get_relevant_documents(query)

    # Tạo instance của GeminiBot
    # gemini_bot = GeminiBot()

    # # Tạo đối tượng RetrievalQA
    # retrieval_qa = RetrievalQA.from_chain_type(
    #     llm = gemini_bot.model,
    #     retriever=retriever,
    #     combine_documents_chain=gemini_bot.response,  # Sử dụng hàm response của GeminiBot
    #     return_source_documents=True
    # )

    # # Trả lời câu hỏi
    # result = retrieval_qa({"query": question})
    # answer = result.get("answer", "Không có câu trả lời nào được tìm thấy.")
    gemini_bot = GeminiBot()
    prompt = """answer my question"""
    ans = gemini_bot.response(prompt+question+str(documents[0]))
    print(documents[0])
    print("="*25)
    return ans

# Ví dụ sử dụng
user_question = "What you will learn in Sentiment Analysis with LSTM and TorchText with Code and Explanation Article?"
answer = answer_question(user_question)
print(answer)
