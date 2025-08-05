"""
Retriever Agent - Handles document retrieval and similarity search
"""

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


class RetrieverAgent:
    """
    Agent responsible for document retrieval and similarity search
    
    Features:
    - Load Chroma vector store from disk
    - Create retriever interface for similarity search
    - Retrieve relevant document chunks for queries
    """
    
    def __init__(self, persist_directory="chroma_db", embedding_model=None):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model or HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )
        self.retriever = self.db.as_retriever()
    
    def retrieve(self, query: str):
        """Retrieve relevant documents for a given query."""
        return self.retriever.get_relevant_documents(query)
