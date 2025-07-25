from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from typing import List, Union
from io import BytesIO
import os
import tempfile
import shutil
import fitz  # PyMuPDF
from langchain.schema import Document

class DocumentProcessorAgent:
    def __init__(self, persist_directory="chroma_db"):
        self.persist_directory = persist_directory
        # Using HuggingFaceEmbeddings directly; adjust if meta tensor errors occur
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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
        docs = []
        temp_dir = tempfile.mkdtemp(prefix="uploaded_docs_")
        try:
            file_paths = []
            for file in uploaded_files:
                if isinstance(file, str):
                    # file is a file path
                    file_paths.append(file)
                else:
                    # file is a BytesIO or UploadedFile
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    file_paths.append(file_path)
            for path in file_paths:
                if path.lower().endswith(".pdf"):
                    # Use PyMuPDF extraction for PDFs
                    docs.extend(self.extract_text_from_pdf(path))
                else:
                    loader = UnstructuredFileLoader(path)
                    docs.extend(loader.load())
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            db = Chroma.from_documents(chunks, self.embedding_model, persist_directory=self.persist_directory)
            db.persist()
        finally:
            shutil.rmtree(temp_dir)
        return db

class RetrieverAgent:
    def __init__(self, persist_directory="chroma_db", embedding_model=None):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model or HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding_model)
        self.retriever = self.db.as_retriever()

    def retrieve(self, query: str):
        return self.retriever.get_relevant_documents(query)

class AnswererAgent:
    def __init__(self, retriever, model_name="llama3:8b", temperature=0.2):
        # Using local llama3:8b model via Ollama
        self.llm = Ollama(model=model_name, temperature=temperature)
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever, return_source_documents=True)

    def answer(self, query: str):
        result = self.qa_chain(query)
        return result
