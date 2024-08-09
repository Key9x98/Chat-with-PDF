from chat import gemini_bot
from chat import set_custom_prompt
import streamlit as st
import time
from pdf_processor import ContextRetriever
from pdf_processor import PDFDatabaseManager

retriever = ContextRetriever("original_text")

class chatBotMode:
    def __init__(self):
        self.mode = "chat" #Mặc định là chế độ chat thông thường
    
    def set_mode(self, mode):
        self.mode = mode

    def process_question(self, user_question):
        if self.mode == "chat":
            return gemini_bot.response(user_question).strip()
        if self.mode == "pdf_query":
            docs = st.session_state.vector_db.similarity_search(user_question, k=2)
            contexts = [doc.page_content for doc in docs]
            metadatas = [doc.metadata for doc in docs]
            expanded_contexts = []
            for i in range(len(contexts)):
                file_name = retriever.get_file_name(metadatas[i])
                print(f"Tên file là: {file_name}")

                expanded_context = retriever.expand_context(file_name, contexts[i])
                expanded_contexts.append(expanded_context)

            context = "\n".join(expanded_contexts)

            prompt = set_custom_prompt()
            st.session_state.history_global.append(user_question + context)
            history_global_str = "\n".join(st.session_state.history_global)
            prompt_with_context = prompt.format(history_global=history_global_str, context=context,
                                                    question=user_question)
            response = gemini_bot.response(prompt_with_context)
            return response, context