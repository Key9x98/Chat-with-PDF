import streamlit as st
import tempfile
import os
from chat import set_custom_prompt
from chat import gemini_bot
from pdf_processor import PDFDatabaseManager
import shutil
import time

vector_db_path = "vectorstores/db_faiss"
hash_store_path = "vectorstores/hashes.json"
pdf_data_path = ''

manager = PDFDatabaseManager(pdf_data_path, vector_db_path, hash_store_path)

def main():
    # Cấu hình trang
    st.set_page_config(page_title="ChatPDF", page_icon='🤖')
    st.header("Vietnamese PDF Chat")

    # Giao diện người dùng cho việc nhập câu hỏi
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

            # Sử dụng gemini_bot để xác định ý định
            intent_prompt = (
                f"Phân loại câu hỏi sau: '{user_question}'. "
                f"Loại nào trong số sau: 'hỏi về câu hỏi trước', 'hỏi về câu trả lời trước', hoặc 'câu hỏi mới'."
            )
            intent_response = gemini_bot.response(intent_prompt).strip().lower()

            if "hỏi về câu hỏi trước" in intent_response:
                last_question = get_last_message(role="user")
                response = f'Bạn vừa hỏi: "{last_question}"' if last_question else "Không có câu hỏi nào được tìm thấy."
            elif "hỏi về câu trả lời trước" in intent_response:
                last_answer = st.session_state.last_answer
                response = f'Tôi vừa trả lời rằng: "{last_answer}"' if last_answer else "Không có câu trả lời nào được tìm thấy."

            # Xử lý câu hỏi mới
            else:
                docs = st.session_state.vector_db.similarity_search(user_question, k=2)
                context = "\n\n".join([doc.page_content for doc in docs])

                prompt = set_custom_prompt()
                st.session_state.history_global.append(user_question + context)
                history_global_str = "\n".join(st.session_state.history_global)
                prompt_with_context = prompt.format(history_global=history_global_str, context=context, question=user_question)
                response = gemini_bot.response(prompt_with_context)

                # Lưu trữ câu trả lời cuối cùng
                st.session_state.last_answer = response

                with st.chat_message('assistant'):
                    st.markdown(response)
                    with st.expander("Show Context", expanded=False):
                        st.write(context)

            # Hiển thị câu trả lời cho các trường hợp không phải câu hỏi mới
            if "hỏi về câu hỏi trước" in intent_response or "hỏi về câu trả lời trước" in intent_response:
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