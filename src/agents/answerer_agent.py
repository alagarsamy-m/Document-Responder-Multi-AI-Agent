"""
Answerer Agent - Handles question answering with local LLaMA model
"""

from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


class AnswererAgent:
    """
    Agent responsible for question answering using local LLaMA 3-8B model
    
    Features:
    - Interface with local LLaMA 3-8B via Ollama
    - Run RetrievalQA chain for RAG (Retrieval-Augmented Generation)
    - Generate contextual answers from retrieved documents
    """
    
    def __init__(self, retriever, model_name="llama3:8b", temperature=0.2):
        # Using local llama3:8b model via Ollama
        self.llm = Ollama(
            model=model_name,
            temperature=temperature
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True
        )
    
    def answer(self, query: str):
        """Generate answer for a given query using retrieved documents."""
        result = self.qa_chain(query)
        return result
