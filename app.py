import streamlit as st
import os
from typing import List, Union
from io import BytesIO
from agents import DocumentProcessorAgent, RetrieverAgent, AnswererAgent
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Document Responder AI", page_icon="🤖", layout="wide")

# Centered big title
st.markdown("<h1 style='text-align: center;'>DOCUMENT RESPONDER AI 🤖</h1>", unsafe_allow_html=True)

# Sidebar with option menu
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Chat", "About", "Requirements", "Project Workflow"],
        icons=["chat-dots", "info-circle", "list-task", "diagram-3"],
        menu_icon="cast",
        default_index=0,
    )

def about_section():
    st.header("About")
    st.write("""
    DOCUMENT RESPONDER AI is a multi-agent AI system that allows you to upload documents and ask questions about their content.
    It uses a local LLaMA 3–8B model via Ollama and Langchain for document processing and question answering.
    """)

def project_workflow_section():
    st.header("Project Workflow")
    st.write("""
    1. Upload documents (PDF, DOCX, TXT).
    2. Documents are processed, chunked, and indexed.
    3. Ask questions via chat interface.
    4. Answers are generated from relevant document chunks using a multi-agent system.
    """)

def requirements_section():
    st.header("Requirements")
    st.write("""
    - Python 3.8+
    - Streamlit
    - Langchain and langchain-community
    - Ollama with LLaMA 3–8B model
    - ChromaDB
    - Other dependencies listed in requirements.txt
    """)

def chat_section():
    st.header("Chat with Document Responder AI")
    # Initialize agents if not in session state
    if "document_processor" not in st.session_state:
        st.session_state.document_processor = DocumentProcessorAgent()
    if "retriever_agent" not in st.session_state:
        st.session_state.retriever_agent = None
    if "answerer_agent" not in st.session_state:
        st.session_state.answerer_agent = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Handle file uploads
    st.subheader("📤 Upload Document(s)")
    uploaded_files: List[Union[BytesIO, str]] = st.file_uploader(
        "Upload .pdf, .docx, or .txt files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        docs_folder = "docs"
        if not os.path.exists(docs_folder):
            os.makedirs(docs_folder)
        for uploaded_file in uploaded_files:
            file_path = os.path.join(docs_folder, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("✅ Uploaded successfully!")
        # Process documents and build vector store from docs folder
        db = st.session_state.document_processor.load_and_process([os.path.join(docs_folder, f) for f in os.listdir(docs_folder)])
        st.session_state.retriever_agent = RetrieverAgent()
        st.session_state.answerer_agent = AnswererAgent(st.session_state.retriever_agent.retriever)

    # Chat UI
    st.subheader("💬 Ask Questions About Your Document")
    query = st.chat_input("Ask a question about the uploaded documents...")

    clear_chat = st.button("Clear Chat History")
    if clear_chat:
        st.session_state.chat_history = []
        st.session_state.retriever_agent = None
        st.session_state.answerer_agent = None
        st.success("Chat history cleared. You can start a new conversation now.")

    if query:
        if st.session_state.answerer_agent:
            result = st.session_state.answerer_agent.answer(query)
            st.session_state.chat_history.append(("user", query))
            st.session_state.chat_history.append(("bot", result["result"]))
        else:
            st.warning("Please upload documents first to build the knowledge base.")

    # Display chat history
    for sender, msg in st.session_state.chat_history:
        if sender == "user":
            st.markdown(f"**🧑‍💻 You:** {msg}")
        else:
            st.markdown(f"**🤖 AI:** {msg}")

# Render selected section
if selected == "About":
    about_section()
elif selected == "Project Workflow":
    project_workflow_section()
elif selected == "Requirements":
    requirements_section()
elif selected == "Chat":
    chat_section()
