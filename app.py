import streamlit as st
import tempfile
import os
from chat import set_custom_prompt
from chat import gemini_bot
from pdf_processor import PDFDatabaseManager
import time
from pdf_processor import ContextRetriever
from botMode import chatBotMode
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

    if "chat_bot" not in st.session_state:
        st.session_state.chat_bot = chatBotMode()
    with st.sidebar:
        st.title("Settings")

        #Thêm chế độ bot
        mode = st.radio("Choose mode: ", ("Chat", "PDF Query"))
        st.session_state.chat_bot.set_mode(mode.lower().replace(" ", "_"))


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
        with st.chat_message("user"):
            st.markdown(user_question)

        response = None  # Khởi tạo response với giá trị mặc định
        context = None
        if st.session_state.chat_bot.mode == "pdf_query":

            if 'vector_db' in st.session_state:
                response, context = st.session_state.chat_bot.process_question(user_question)
                with st.chat_message('assistant'):
                    message_placeholder = st.empty()
                    full_response = ""
                    for chunk in response.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    with st.expander("Show Context", expanded=False):
                        st.write(context)
            else:
                st.warning("Hãy đưa file của bạn lên trước, chúng tôi sẽ dựa vào đó để trả lời")
        else:
            response = st.session_state.chat_bot.process_question(user_question)
            with st.chat_message('assistant'):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

        # Thêm tin nhắn của người dùng và phản hồi của trợ lý vào lịch sử chat
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.session_state.messages.append({"role": "assistant", "content": response})
        

if __name__ == "__main__":
    main()
