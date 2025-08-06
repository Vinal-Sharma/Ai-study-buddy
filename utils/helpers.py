# utils/helpers.py

import PyPDF2
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

def get_pdf_text(pdf_file):
    """Extracts text from an uploaded PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error reading PDF file: {e}"

def get_text_chunks(text):
    """Splits text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # The size of each chunk in characters
        chunk_overlap=200, # How many characters to overlap between chunks
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, embedding_model):
    """Creates a FAISS vector store from text chunks."""
    try:
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embedding_model)
        return vector_store
    except Exception as e:
        return f"Error creating vector store: {e}"