import streamlit as st
import tempfile
import os
from chat import set_custom_prompt
from chat import gemini_bot
from pdf_processor import PDFDatabaseManager
import time
from pdf_processor import ContextRetriever

vector_db_path = "vectorstores/db_faiss"
hash_store_path = "vectorstores/hashes.json"
pdf_data_path = ''

manager = PDFDatabaseManager(pdf_data_path, vector_db_path, hash_store_path)
retriever = ContextRetriever("original_text")


def main():
    # Cấu hình trang
    st.set_page_config(page_title="ChatPDF", page_icon='🤖')
    st.header("Vietnamese PDF Chat")
    st.markdown(
        """
        <style>
        .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
        .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
        .viewerBadge_text__1JaDK {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    user_question = st.chat_input("Ask a Question from the PDF Files")

    if "history_global" not in st.session_state:
        st.session_state.history_global = []

    with st.sidebar:
        st.title("Upload PDF")
        pdf_docs = st.file_uploader("You can upload multiple PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    status_placeholder = st.empty()
                    with tempfile.TemporaryDirectory() as temp_dir:
                        manager.pdf_data_path = temp_dir
                        for uploaded_file in pdf_docs:
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                        pdf_files = [f for f in os.listdir(temp_dir) if f.lower().endswith('.pdf')]
                        for pdf_file in pdf_files:
                            file_path = os.path.join(temp_dir, pdf_file)
                            if not manager.is_pdf_exists(file_path):
                                status_placeholder.write(f"Tệp {pdf_file} chưa có trong db. Processing...")
                                manager.update_db(file_path)
                            else:
                                status_placeholder.write(f"Tệp {pdf_file} đã có trong db, gửi tệp khác.")
                    db = manager.load_existing_db()
                    if db is not None:
                        st.session_state.vector_db = db
                        st.success("PDFs processed and vector database created!")

                        time.sleep(3)
                        status_placeholder.empty()

            else:
                st.warning("No file selected")
        st.markdown(
            "Nếu muốn tìm thông tin trong PDF, hãy sử dụng những từ khóa như PDF, tài liệu, file, tài liệu, bài báo ")

    # Khởi tạo lịch sử chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hiển thị tin nhắn chat từ lịch sử trên lần chạy lại ứng dụng
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question:

        if 'vector_db' in st.session_state:
            with st.chat_message("user"):
                st.markdown(user_question)

            # if handle_chat.is_chitchat_question(user_question, handle_chat.chitchatSample):
            #     response = handle_chat.handle_chitchat(user_question)
            #     with st.chat_message('assistant'):
            #         st.markdown(response)

            pdf_related_keywords = ["tài liệu", "pdf", "file", "tệp", "bài báo"]

            if any(keyword in user_question.lower() for keyword in pdf_related_keywords):
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

                # Lưu trữ câu trả lời cuối cùng
                st.session_state.last_answer = response

                with st.chat_message('assistant'):
                    st.markdown(response)
                    with st.expander("Show Context", expanded=False):
                        st.write(context)
            else:
                response = gemini_bot.response(user_question).strip()
                with st.chat_message('assistant'):
                    st.markdown(response)
            # Thêm tin nhắn của người dùng và phản hồi của trợ lý vào lịch sử chat
            st.session_state.messages.append({"role": "user", "content": user_question})
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.warning("Hãy đưa file của bạn lên trước, chúng tôi sẽ dựa vào đó để trả lời")


def get_last_message(role):
    messages = st.session_state.get("messages", [])
    for message in reversed(messages):
        if message["role"] == role:
            return message["content"]
    return None


if __name__ == "__main__":
    main()
