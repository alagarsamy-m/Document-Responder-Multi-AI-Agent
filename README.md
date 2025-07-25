# Document Responder AI

## Overview
Document Responder AI is a multi-agent AI system that allows users to upload documents and ask questions about their content through an interactive web interface. It leverages a local LLaMA 3-8B model accessed via Ollama and Langchain for document processing, embedding, retrieval, and question answering.

## Features
- Upload multiple documents in PDF, DOCX, and TXT formats.
- Documents are processed, chunked, and indexed into a vector store.
- Ask questions about the uploaded documents via a chat interface.
- Answers are generated using a local LLaMA 3-8B model with retrieval-augmented generation.
- Clear chat history to start new conversations.
- Uses PyMuPDF for PDF processing.

## Project Structure
```
.
├── agents.py               # Core agent classes for document processing, retrieval, and answering
├── app.py                  # Streamlit web application UI
├── requirements.txt        # Python dependencies
├── MODEL_USAGE.md          # Instructions for using the local LLaMA model with Ollama
├── README.md               # Project documentation
├── chroma_db/              # Persistent Chroma vector store database files
└── docs/                   # Uploaded documents folder and additional docs
```

## Dependencies / Requirements
- Python 3.8+
- Streamlit
- Langchain and langchain-community
- Ollama with local LLaMA 3-8B model running
- ChromaDB and faiss-cpu for vector storage and similarity search
- sentence-transformers for embedding models
- unstructured[pdf], pypdf, pymupdf for document parsing
- torch
- streamlit-option-menu
- tiktoken

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Local LLaMA Model Setup
- Download the LLaMA 3-8B model files and place them in a `models` directory inside the project.
- Configure Ollama to use the local model path as described in `MODEL_USAGE.md`.
- Ensure Ollama is running locally with access to the model.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open the URL provided by Streamlit in your browser.
4. Upload documents (PDF, DOCX, TXT) via the UI.
5. Ask questions about the uploaded documents in the chat interface.
6. Use the "Clear Chat History" button to reset the conversation.

## Project Workflow
1. Upload documents through the web UI.
2. Documents are loaded and processed:
   - PDFs are processed using PyMuPDF.
   - Other formats are processed using UnstructuredFileLoader.
3. Documents are chunked and embedded using HuggingFace embeddings.
4. Embeddings are stored in a persistent Chroma vector store.
5. User queries are answered by retrieving relevant chunks and generating answers with the local LLaMA model.
6. Chat history is maintained for context and can be cleared anytime.

## Notes
- PDF processing uses PyMuPDF to avoid external dependencies like Poppler.
- Ensure the `models` directory contains the local LLaMA model files for Ollama.
- The system is designed for local, offline use with privacy and speed benefits.

## Screenshots
![Screenshot 2025-07-25 231525](screenshots/Screenshot%202025-07-25%20231525.png)
