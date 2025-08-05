"""
Document Processor Agent - Handles document ingestion and preprocessing
"""

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List, Union
from io import BytesIO
import os
import tempfile
import shutil
import fitz  # PyMuPDF


class DocumentProcessorAgent:
    """
    Agent responsible for document processing, chunking, and vector storage
    
    Features:
    - Extract text from PDF, DOCX, TXT files
    - Split documents into chunks with configurable overlap
    - Generate embeddings using HuggingFace models
    - Store in persistent Chroma vector database
    """
    
    def __init__(self, persist_directory="chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    
    def extract_text_from_pdf(self, file_path: str) -> List[Document]:
        """Extract text from PDF using PyMuPDF and return as Langchain Documents."""
        docs = []
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            metadata = {"source": file_path, "page": page_num + 1}
            docs.append(Document(page_content=text, metadata=metadata))
        return docs
    
    def load_and_process(self, uploaded_files: List[Union[str, BytesIO]]):
        """Process uploaded documents and build vector store."""
        docs = []
        temp_dir = tempfile.mkdtemp(prefix="uploaded_docs_")
        try:
            file_paths = []
            for file in uploaded_files:
                if isinstance(file, str):
                    file_paths.append(file)
                else:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    file_paths.append(file_path)
            
            for path in file_paths:
                if path.lower().endswith(".pdf"):
                    docs.extend(self.extract_text_from_pdf(path))
                else:
                    loader = UnstructuredFileLoader(path)
                    docs.extend(loader.load())
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = splitter.split_documents(docs)
            
            db = Chroma.from_documents(
                chunks,
                self.embedding_model,
                persist_directory=self.persist_directory
            )
            db.persist()
            return db
        finally:
            shutil.rmtree(temp_dir)
