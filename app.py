import streamlit as st
import tempfile
import os
from chat import set_custom_prompt
from chat import gemini_bot
from pdf_processor import PDFDatabaseManager
import time
from pdf_processor import ContextRetriever
from botMode import chatBotMode
from text_processor import TextProcessor

text_processor = TextProcessor()

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
                display_response = text_processor.remove_markdown(response)
                with st.chat_message('assistant'):
                    message_placeholder = st.empty()
                    full_response = ""
                    for chunk in display_response.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.write(full_response + "▌", unsafe_allow_html=True)
                    time.sleep(0.1)
                    final_message = "**_Câu trả lời trích từ tài liệu:_**\n\n" + response
                    message_placeholder.markdown(final_message, unsafe_allow_html=True)
                    with st.expander("Show Context", expanded=False):
                        st.write(context)
            else:
                st.warning("Hãy đưa file của bạn lên trước, chúng tôi sẽ dựa vào đó để trả lời")
        else:
            response = st.session_state.chat_bot.process_question(user_question)
            display_response = text_processor.remove_markdown(response)
            with st.chat_message('assistant'):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in display_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.write(full_response + "▌", unsafe_allow_html=True)
                time.sleep(0.1)
                message_placeholder.markdown(response + "\n", unsafe_allow_html=True)

        # Thêm tin nhắn của người dùng và phản hồi của trợ lý vào lịch sử chat
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.session_state.messages.append({"role": "assistant", "content": response})

    if st.session_state.messages:
        last_message = st.session_state.messages[-1]
        end_punctuation = {'.', '!', '?'}


        if last_message["role"] == "assistant" and not any(
                last_message["content"].strip().endswith(punct) for punct in end_punctuation):

            # Tạo nút "Generate more" bên ngoài chat_message
            generate_more = st.button("Generate more", key=f"generate_more_{len(st.session_state.messages)}")

            if generate_more:
                continue_prompt = f"""
                                        Câu trả lời trước: {last_message['content']}
                                        Câu hỏi: {user_question}
                                        Context: {context}
                                        Hãy tiếp tục trả lời từ phần hiện tại, không trả lời lặp lại phần trước,
                                        Nếu có yêu cầu về số từ, số từ = số từ trong câu trả lời trước + số từ trong câu trả lời mới:
                                    """
                continuation = st.session_state.chat_bot.process_question(continue_prompt)
                display_continuation = text_processor.remove_markdown(continuation)

                with st.container():
                    message_placeholder = st.empty()
                    full_continuation = ""
                    for chunk in display_continuation.split():
                        full_continuation += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.write(full_continuation + "▌", unsafe_allow_html=True)
                    time.sleep(0.1)
                    message_placeholder.markdown(continuation + "\n", unsafe_allow_html=True)

                st.session_state.messages.append({"role": "assistant", "content": continuation})

if __name__ == "__main__":
    main()
