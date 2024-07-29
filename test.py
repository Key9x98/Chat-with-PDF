import google.generativeai as genai
import os
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from embedding import custom_embeddings

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
    "max_output_tokens": 1024*3,
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
        # self.token_count = self.model.count_tokens(PROMPT).total_tokens
        self.chat = self.model.start_chat()

    def response(self, user_input):
        # input_tokens = self.model.count_tokens(user_input).total_tokens
        # if self.token_count > 1000000:
        #     return "Out of tokens"
        response = self.chat.send_message(user_input)
        # self.token_count += input_tokens + self.model.count_tokens(response.text).total_tokens
        return response.text

# Đọc cơ sở dữ liệu vector
def read_vectors_db():
    embedding_model = custom_embeddings
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

custom_prompt_template = """Bạn là một hệ thống hỏi đáp, nhiệm vụ là tổng hợp thông tin trong Context để trả lời câu hỏi
1. Nếu câu trả lời không có trong Context hoặc bạn không chắc chắn, hãy trả lời "Tôi không có đủ thông tin để trả lời câu hỏi này."
2. Không suy đoán và bịa đặt nội dung ngoài
3. Trả lời đầy đủ thông tin theo Context tìm được
4. Sử dụng tiếng việt
5. Nếu người dùng yêu cầu giải thích hoặc tương tự, hãy tìm kiếm thông tin bên ngoài bổ sung thêm và trả lời

Context: {context}
Question: {question}

Câu trả lời:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Tạo hàm để trả lời câu hỏi dựa vào cơ sở dữ liệu vector
def answer_question(question):
    # Đọc cơ sở dữ liệu vector
    db = read_vectors_db()
    retriever = db.as_retriever()

    # Retrieve relevant documents
    documents = retriever.invoke(question)

    # Create context from documents
    context = "\n\n".join([doc.page_content for doc in documents])
    print("="*25)
    print(context)
    # print("="*25)
    # Set custom prompt
    custom_prompt = set_custom_prompt()
    prompt_context = custom_prompt.format(context=context, question=question)

    gemini_bot = GeminiBot()
    answer = gemini_bot.response(prompt_context)
    return answer

# Ví dụ sử dụng
user_question = "Các trường hợp sinh viên được tạm nghỉ học"
answer = answer_question(user_question)
print(answer)
print("="*25)
