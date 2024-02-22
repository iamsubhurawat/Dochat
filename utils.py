from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

import os
import tempfile
from typing import List
from tqdm import tqdm

def create_llm():
    """
    Create an instance of Mistral 7B GCUF format LLM using LLamaCpp

    :return:
    - llm: An instance of Mistral 7B LLM
    """
    # Create llm
    llm = LlamaCpp(
        streaming=True,
        model_path="artifcats/mistral-7b-instruct-v0.1.Q5_0.gguf",
        temperature=0.3,
        top_p=0.8,
        verbose=True,
        n_ctx=4096 #Context Length
    )
    return llm

def create_vector_store(pdf_files: List):
    """
    Create In-memory FAISS vector store using uploaded Pdf

    Args:
    - pdf_files(List): PDF file uploaded
    :return:
    - vector_store: In-memory Vector store for further processing at chat app
    """
    vector_store = None

    if pdf_files:
        text = []

        for file in tqdm(pdf_files, desc="Processing files"):
            # Get the file and check it's extension
            file_extension = os.path.splitext(file.name)[1]
            # Write the PDF file to temporary location
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            # Load the PDF files using PyPdf library
            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)

            # Load if text file
            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        # Split the file to chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=10)
        text_chunks = text_splitter.split_documents(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={'device': 'cpu'})

        # Create vector store and storing document chunks using embedding model
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    return vector_store

