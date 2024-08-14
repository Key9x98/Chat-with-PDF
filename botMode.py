from chat import gemini_bot
from chat import set_custom_prompt
import streamlit as st
import time
from pdf_processor import ContextRetriever
from text_processor import TextProcessor

from pdf_processor import PDFDatabaseManager
from langchain_community.vectorstores import FAISS
from operator import itemgetter

retriever = ContextRetriever("original_text")
text_processor = TextProcessor()


class chatBotMode:
    def __init__(self):
        self.mode = "chat"  # Mặc định là chế độ chat thông thường

    def set_mode(self, mode):
        self.mode = mode

    def process_question(self, user_question):
        if self.mode == "chat":
            return gemini_bot.response(user_question).strip()
        if self.mode == "pdf_query":
            # Chỉ tìm kiếm trong các vector_db đã được chọn
            selected_dbs = {name: db for name, db in self.vector_db.items() if name in st.session_state.selected_pdfs}

            if not selected_dbs:
                return "Vui lòng chọn ít nhất một tài liệu PDF để truy vấn.", ""

            # Tìm trong các vector đã chọn (chưa song song)
            results = []
            for db_name, db in selected_dbs.items():
                docs_scores = db.similarity_search_with_score(user_question, k=2)
                results.extend([(doc, score, db_name) for doc, score in docs_scores])

            # rank by score, lấy 2 best
            top_docs = sorted(results, key=itemgetter(1))[:2]

            contexts = []
            metadatas = []
            expanded_contexts = []

            for doc, score, db_name in top_docs:
                file_name = retriever.get_file_name(doc.metadata)
                print(f"Tên file là: {file_name}, Điểm số: {score}")

                doc.metadata['source_db'] = db_name
                contexts.append(doc.page_content)
                metadatas.append(doc.metadata)

                # Vì file name.txt đã xóa dấu nên remove cả ở expander

                file_name_remove_accents = text_processor.remove_accents(file_name)

                expanded_context = retriever.expand_context(file_name_remove_accents, doc.page_content)
                expanded_contexts.append(expanded_context)

            context = "\n".join(expanded_contexts)

            prompt = set_custom_prompt()

            # Giới hạn lịch sử để tránh quá tải
            # st.session_state.history_global = st.session_state.history_global[-5:]

            st.session_state.history_global.append(user_question + context)
            history_global_str = "\n".join(st.session_state.history_global[-1])

            prompt_with_context = prompt.format(
                history_global=history_global_str,
                context=context,
                question=user_question
            )

            response = gemini_bot.response(prompt_with_context)

            # Thêm thông tin về nguồn
            sources = [f"{meta['source_db']} ({retriever.get_file_name(meta)})" for meta in metadatas]
            response_with_sources = f"Nguồn: {', '.join(set(sources))}. \n\n {response}"

            return response_with_sources, context