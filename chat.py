import google.generativeai as genai
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

# Load .env file
load_dotenv()

# GOOGLE_API_KEY = os.getenv("API_KEY")
# MODEL_NAME =  os.getenv("MODEL_NAME")

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
MODEL_NAME = st.secrets["MODEL_NAME"]


print(MODEL_NAME)

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
# Chuyển đổi mảng thành chuỗi với các tin nhắn được nối nhau

custom_prompt_template = """Bạn là một hệ thống hỏi đáp, nhiệm vụ là tổng hợp thông tin trong Context để trả lời câu hỏi
1. Nếu câu trả lời không có trong Context hoặc bạn không chắc chắn, hãy trả lời "Tôi không có đủ thông tin để trả lời câu hỏi này."
2. Không suy đoán và bịa đặt nội dung ngoài
3. Chỉ trả lời thông tin theo Context tìm được, một cách đầy đủ 
4. Sử dụng tiếng việt

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

class GeminiBot:
  def __init__(self):
    self.model_name = MODEL_NAME
    self.token_count = None
    self.model = None
    self.chat = None
    self._setup()

  def _setup(self):
    genai.configure(api_key=GOOGLE_API_KEY)
    PROMPT = 'Ban là một công cụ chat, hãy trả lời các câu hỏi của người dùng'
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


gemini_bot = GeminiBot()

# Sử dụng đối tượng để trả lời câu hỏi
user_question = "1 + 1 bằng mấy"
bot_response = gemini_bot.response(user_question)
print(bot_response)
