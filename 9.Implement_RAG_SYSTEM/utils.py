import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
pdf_path = "assets/mypdf.pdf"

def load_and_split_pdf():
    loader = PyPDFLoader(pdf_path)
    return loader.load_and_split()

def build_or_load_vectorstore():
    if os.path.exists("vectorstore/index.faiss"):
        return FAISS.load_local("vectorstore", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    else:
        docs = load_and_split_pdf()
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectordb = FAISS.from_documents(docs, embeddings)
        vectordb.save_local("vectorstore")
        return vectordb