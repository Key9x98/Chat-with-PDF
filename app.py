import streamlit as st
import tempfile
import os
from pdf_processor import PDFDatabaseManager
import time
from pdf_processor import ContextRetriever
from botMode import chatBotMode
from text_processor import TextProcessor
from chat import gemini_bot

vector_db_path = "vectorstores/db_faiss"
hash_store_path = "vectorstores/hashes.json"
pdf_data_path = ''

text_processor = TextProcessor()
manager = PDFDatabaseManager(pdf_data_path, vector_db_path, hash_store_path)
retriever = ContextRetriever("original_text")

def hide_elements():
    hide_elements_style = """
    <style>
    /* Hide the GitHub icon */
    .viewerBadge_container__1QSob {
        display: none !important;
    }
    /* Hide the header */
    header {
        visibility: hidden !important;
    }
    /* Hide the footer */
    footer {
        visibility: hidden !important;
    }
    </style>
    """
    st.markdown(hide_elements_style, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="ChatPDF", page_icon='🤖')
    col1, col2 = st.columns([0.6, 0.4])  # Tạo hai cột với tỉ lệ 9:1

    with col1:
        st.header("Vietnamese PDF Chat")

    with col2:
        if st.button("🧹", help="Clean data and reload"):
            clean_data()

    hide_elements()

    user_question = st.chat_input("Ask a Question from the PDF Files")



    if "history_global" not in st.session_state:
        st.session_state.history_global = []
    if "chat_bot" not in st.session_state:
        st.session_state.chat_bot = chatBotMode()
    if "processed_pdfs" not in st.session_state:
        st.session_state.processed_pdfs = []
    if "selected_pdfs" not in st.session_state:
        st.session_state.selected_pdfs = set()
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = {}

    st.session_state.chat_bot.vector_db = st.session_state.vector_db

    with st.sidebar:
        st.title("Settings")

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
                                status_placeholder.write(f"Processing {pdf_file}...")
                                db = manager.update_db(file_path)
                                if db is not None:
                                    st.session_state.vector_db[pdf_file] = db
                                st.session_state.processed_pdfs.append(pdf_file)
                                st.session_state.selected_pdfs.add(pdf_file)
                            else:
                                status_placeholder.write(f"{pdf_file} already exists in the database.")

                    for pdf_file in st.session_state.processed_pdfs:
                        db = manager.load_existing_db(pdf_file)
                        if db is not None:
                            st.session_state.vector_db[pdf_file] = db

                    st.success("PDFs processed and vector database created!")
                    time.sleep(2)
                    status_placeholder.empty()
            else:
                st.warning("No file selected")

        st.title("Manage PDFs")
        if st.session_state.chat_bot.mode == "pdf_query":
            if not st.session_state.vector_db:
                st.info("No PDFs uploaded yet. Please upload PDFs to query.")
            else:
                pdfs_status_changed = False
                for pdf in st.session_state.vector_db:
                    selected = st.checkbox(f"Query {pdf}", value=pdf in st.session_state.selected_pdfs)
                    if selected != (pdf in st.session_state.selected_pdfs):
                        pdfs_status_changed = True
                    if selected:
                        st.session_state.selected_pdfs.add(pdf)
                    else:
                        st.session_state.selected_pdfs.discard(pdf)
                if pdfs_status_changed:
                    st.success("PDF selection updated.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)

        response = None
        context = None
        if st.session_state.chat_bot.mode == "pdf_query":
            if st.session_state.vector_db:
                selected_dbs = {pdf: st.session_state.vector_db[pdf] for pdf in st.session_state.selected_pdfs if
                                pdf in st.session_state.vector_db}
                if selected_dbs:
                    temp_vector_db = st.session_state.chat_bot.vector_db
                    st.session_state.chat_bot.vector_db = selected_dbs
                    
                    with st.spinner("🧐🧐🧐 thinking..."):
                        response, context = st.session_state.chat_bot.process_question(user_question)

                    st.session_state.chat_bot.vector_db = temp_vector_db

                    display_response = text_processor.remove_markdown(response)
                    with st.chat_message('assistant'):
                        message_placeholder = st.empty()
                        full_response = ""
                        for chunk in display_response.split():
                            full_response += chunk + " "
                            time.sleep(0.04)
                            message_placeholder.write(full_response + "▌")
                        time.sleep(0.1)
                        final_message = "**Answer extracted from the document:**\n\n" + response
                        message_placeholder.markdown(final_message)
                        with st.expander("Show Context", expanded=False):
                            formatted_context = text_processor.format_context(context)
                            st.markdown(formatted_context, unsafe_allow_html=True)

                        st.session_state.current_context = context
                else:
                    st.warning("Please select at least one PDF to query.")
            else:
                st.warning("Please upload your files first, we will use them to answer.")
        else:
            st.session_state.current_context = ""
            with st.spinner("🧐🧐🧐 thinking..."):
                response = st.session_state.chat_bot.process_question(user_question)
            display_response = text_processor.remove_markdown(response)
            with st.chat_message('assistant'):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in display_response.split():
                    full_response += chunk + " "
                    time.sleep(0.04)
                    message_placeholder.write(full_response + "▌")
                time.sleep(0.1)
                message_placeholder.markdown(response + "\n")

        st.session_state.messages.append({"role": "user", "content": user_question})
        st.session_state.messages.append({"role": "assistant", "content": response})

    if st.session_state.messages:
        last_message = st.session_state.messages[-1]
        end_punctuation = text_processor.get_end_tokens()

        if (last_message["role"] == "assistant" and
                last_message.get("content") and
                not any(last_message["content"].strip().endswith(punct) for punct in end_punctuation)):

            # Tạo nút "Generate more" bên ngoài chat_message
            generate_more = st.button("Generate more", key=f"generate_more_{len(st.session_state.messages)}")

            if generate_more:
                context = st.session_state.current_context
                continue_prompt = f"""
                                            Câu trả lời trước: {last_message['content']},
                                            Câu hỏi: {user_question},
                                            Context: {context},
                                            Hãy tiếp tục trả lời từ phần hiện tại sang phải, không trả lời lặp lại phần trước.
                                            Context có thể có hoặc không.
                                            Nếu context xuất hiện, chỉ trả lời theo context, không thêm nội dung ngoài.
                                            Nếu không có yêu cầu về số từ, hãy dừng lại khi câu trả lời đầy đủ.
                                            Nếu có yêu cầu về số từ, số từ = số từ trong câu trả lời trước + số từ trong câu trả lời mới:
                                        """
                continuation = gemini_bot.response(continue_prompt)
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

import shutil
def clean_data():
    try:
        if os.path.exists("vectorstores"):
            shutil.rmtree("vectorstores")
            
        if os.path.exists("original_text"):
            shutil.rmtree("original_text")

        for key in list(st.session_state.keys()):
            del st.session_state[key]

        st.warning("Dữ liệu đã được xóa. Trang sẽ tải lại sau 3 giây...")
        time.sleep(3)

        st.rerun()
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi dọn dẹp dữ liệu: {str(e)}")

if __name__ == "__main__":
    main()
