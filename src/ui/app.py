"""
Main Streamlit Application - Main entry point for the Document Responder AI
"""

import streamlit as st
import os
from typing import List, Union
from io import BytesIO
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.agents.document_processor_agent import DocumentProcessorAgent
from src.agents.retriever_agent import RetrieverAgent
from src.agents.answerer_agent import AnswererAgent


class DocumentResponderApp:
    """Main application class for Document Responder AI"""
    
    def __init__(self):
        st.set_page_config(
            page_title="Document Responder AI",
            page_icon="🤖",
            layout="wide"
        )
        
    def run(self):
        """Main application runner"""
        st.markdown(
            "<h1 style='text-align: center;'>DOCUMENT RESPONDER AI 🤖</h1>",
            unsafe_allow_html=True
        )
        
        # Sidebar navigation
        with st.sidebar:
            selected = st.selectbox(
                "Navigation",
                ["Chat", "About", "Requirements", "Project Workflow"]
            )
        
        if selected == "Chat":
            self._render_chat_section()
        elif selected == "About":
            self._render_about_section()
        elif selected == "Requirements":
            self._render_requirements_section()
        elif selected == "Project Workflow":
            self._render_workflow_section()
    
    def _render_chat_section(self):
        """Render the chat interface"""
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
        uploaded_files = st.file_uploader(
            "Upload .pdf, .docx, or .txt files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
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
            
            # Process documents
            file_paths = [os.path.join(docs_folder, f) for f in os.listdir(docs_folder)]
            db = st.session_state.document_processor.load_and_process(file_paths)
            
            st.session_state.retriever_agent = RetrieverAgent()
            st.session_state.answerer_agent = AnswererAgent(
                st.session_state.retriever_agent.retriever
            )
        
        # Chat interface
        st.subheader("💬 Ask Questions About Your Document")
        query = st.chat_input("Ask a question about the uploaded documents...")
        
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
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.retriever_agent = None
            st.session_state.answerer_agent = None
            st.success("Chat history cleared. You can start a new conversation now.")
    
    def _render_about_section(self):
        """Render the about section"""
        st.header("About Document Responder AI")
        st.write("""
        **Document Responder AI** is a cutting-edge multi-agent AI system that revolutionizes how you interact with documents. 
        
        ### Key Features:
        - 📄 **Multi-format Support**: PDF, DOCX, TXT files
        - 🤖 **Local AI**: Uses LLaMA 3-8B model via Ollama
        - 🔍 **Smart Retrieval**: Context-aware document search
        - 💬 **Conversational Interface**: Natural language queries
        - 🔒 **Privacy-First**: All processing happens locally
        
        ### Technology Stack:
        - **Backend**: Python 3.8+, LangChain, ChromaDB
        - **Frontend**: Streamlit
        - **AI Model**: LLaMA 3-8B via Ollama
        - **Embeddings**: HuggingFace all-MiniLM-L6-v2
        
        This system is designed for researchers, students, legal professionals, and anyone who needs to efficiently extract insights from documents.
        """)
    
    def _render_requirements_section(self):
        """Render the requirements section"""
        st.header("System Requirements")
        
        st.write("""
        ### Software Requirements:
        - **Python**: 3.8 or higher
        - **Ollama**: Latest version with LLaMA 3-8B model
        - **Operating System**: Windows 10/11, macOS, or Linux
        
        ### Python Dependencies:
        ```
        langchain                    # AI orchestration framework
        langchain-community          # Community integrations
        sentence-transformers        # Embedding models
        chromadb                    # Vector database
        faiss-cpu                   # Similarity search
        unstructured[pdf]           # Document parsing
        pymupdf                     # PDF processing
        python-docx                 # DOCX processing
        streamlit                   # Web UI framework
        tiktoken                    # Token counting
        streamlit-option-menu       # UI components
        torch                       # PyTorch for embeddings
        ```
        
        ### Installation Steps:
        1. **Install Python 3.8+** from python.org
        2. **Install Ollama** from https://ollama.ai
        3. **Download LLaMA 3-8B model**:
           ```bash
           ollama pull llama3:8b
           ```
        4. **Install Python dependencies**:
           ```bash
           pip install -r requirements.txt
           ```
        """)
    
    def _render_workflow_section(self):
        """Render the project workflow section"""
        st.header("Project Workflow")
        
        st.write("""
        ### Document Processing Workflow:
        
        **1. Document Upload**
        - User uploads documents via web interface
        - Files are stored in `docs/` folder
        
        **2. Text Extraction**
        - PDF: Extracted using PyMuPDF (fitz)
        - DOCX: Processed with python-docx
        - TXT: Direct text processing
        
        **3. Chunking & Embedding**
        - Split into 500-character chunks with 50-char overlap
        - Generate embeddings using HuggingFace all-MiniLM-L6-v2
        - Store in Chroma vector database
        
        **4. Query Processing**
        - User asks questions via chat interface
        - Retrieve relevant chunks using similarity search
        - Generate contextual answers using LLaMA 3-8B
        
        **5. Response Delivery**
        - Display answers in conversational format
        - Maintain chat history for context
        
        ### Query Processing Pipeline:
        1. **Receive Query** → User question via chat
        2. **Retrieve Context** → Similarity search in vector store
        3. **Augment Query** → Combine with relevant chunks
        4. **Generate Answer** → LLaMA 3-8B creates response
        5. **Display Result** → Show answer in chat interface
        
        ### Performance Metrics:
        - **Response Time**: 2-5 seconds per query
        - **Supported Formats**: PDF, DOCX, TXT
        - **Chunk Size**: 500 characters
        - **Embedding Dimensions**: 384
        - **Model**: LLaMA 3-8B (8B parameters)
        """)


# Main execution
if __name__ == "__main__":
    app = DocumentResponderApp()
    app.run()
