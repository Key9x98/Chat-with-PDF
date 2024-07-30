import streamlit as st
from document_processor import DocumentProcessor
import tempfile
import os
from chat import set_custom_prompt
from chat import gemini_bot


def main():
    # Cấu hình trang
    st.set_page_config(page_title="ChatPDF", page_icon='🤖')
    st.header("Vietnamese PDF Chat")

    # Giao diện người dùng cho việc nhập câu hỏi
    user_question = st.chat_input("Ask a Question from the PDF Files")

    # Biến đánh dấu xem có tệp nào đã được tải lên không
    pdf_docs = None
    history_global = []
    with st.sidebar:
        st.title("Upload PDF")
        pdf_docs = st.file_uploader("You can upload multiple PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Save uploaded files to the temporary directory
                        for uploaded_file in pdf_docs:
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                        processor = DocumentProcessor(temp_dir, "vectorstores/db_faiss")
                        db = processor.run()
                    st.session_state.vector_db = db
                    st.success("PDFs processed and vector database created!")
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
        # Kiểm tra xem có tệp nào đã được tải lên chưa
        if 'vector_db' in st.session_state:
            # Hiển thị tin nhắn của người dùng trong hộp tin nhắn chat
            with st.chat_message("user"):
                st.markdown(user_question)

            docs = st.session_state.vector_db.similarity_search(user_question, k=2)
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = set_custom_prompt()
            history_global.append(user_question + context)
            history_global_str = "\n".join(history_global)
            prompt_with_context = prompt.format(history_global= history_global_str, context=context, question=user_question)
            response = gemini_bot.response(prompt_with_context)


            # Hiển thị phản hồi trong hộp tin nhắn chat
            with st.chat_message('assistant'):
                st.markdown(response)

            # Thêm tin nhắn của người dùng vào lịch sử chat
            st.session_state.messages.append({"role": "user", "content": user_question})
            # Thêm phản hồi của trợ lý vào lịch sử chat
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.warning("Hãy đưa file của bạn lên trước, chúng tôi sẽ dựa vào đó để trả lời")

if __name__ == "__main__":
    main()
