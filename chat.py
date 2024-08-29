import google.generativeai as genai
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Chạy Local
# Cách viết .env: xem file .envexalmple
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("API_KEY")
# MODEL_NAME =  os.getenv("MODEL_NAME")

# # Chạy trên Streamlit: thêm 2 trường ở config
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
MODEL_NAME = st.secrets["MODEL_NAME"]

generation_config = {
  "temperature": 0.05,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 1000,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
]

custom_prompt_template = """
Yêu cầu cụ thể là tổng hợp thông tin trong các đoạn Context để trả lời câu hỏi. 
Các chỉ mục đánh số dưới đây là các mô tả nhiệm vụ:

1. Nếu câu trả lời không có trong Context hoặc bạn không chắc chắn, hãy trả lời: 
   "Tôi không có đủ thông tin để trả lời câu hỏi này. Vui lòng cung cấp thêm thông tin liên quan đến câu hỏi."

2. Không suy đoán và bịa đặt nội dung ngoài.

3. Chỉ trả lời thông tin theo Context tìm được. Trả lời đầy đủ thông tin liên quan đến câu hỏi, 
   bao gồm cả việc liệt kê các ý nhỏ nếu cần.

4. Thông tin thường chỉ nằm trong một đoạn context. Các đoạn context được chia cách bởi chuỗi "SEPARATED".

5. Chỉ sử dụng History khi người dùng hỏi về câu hỏi trước đó:

History: {history_global}

Context: {context}

Question: {question}

Câu trả lời:
"""

def set_custom_prompt():
  prompt = PromptTemplate(template=custom_prompt_template,
                          input_variables=['history_global','context', 'question'])
  return prompt

class GeminiBot:
  def __init__(self):
    self.model_name = MODEL_NAME
    self.token_count = None
    self.model = None
    self.chat = None
    self._setup()

  def _setup(self):
    genai.configure(api_key=GOOGLE_API_KEY)
    INSTRUCTION = ('Bạn là một công cụ hỏi đáp, hãy trả lời các câu hỏi của người dùng bằng tiếng Việt.'
              'Khi kết thúc trả lời hãy hỏi "\n\n Nếu bạn cần thêm điều gì, hãy cho tôi biết!" hoặc những câu tương tự.'
              'Người dùng sẽ có thêm những mô tả cụ thể, hãy làm theo những yêu cầu đó.')
    self.model = genai.GenerativeModel(model_name=self.model_name,
                                       generation_config=generation_config,
                                       safety_settings=safety_settings,
                                       system_instruction=INSTRUCTION)
    self.chat = self.model.start_chat()


  def response(self, user_input):
    user_input = user_input
    response = self.chat.send_message(user_input)
    return response.text


gemini_bot = GeminiBot()
