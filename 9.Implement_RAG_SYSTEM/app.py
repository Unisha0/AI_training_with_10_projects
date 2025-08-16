import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import build_or_load_vectorstore

load_dotenv()
st.set_page_config(page_title=" 🇳🇵 Chat with CHATBOT ABOUT NEPAL 🇳🇵 ")

st.title(" Ask Anything About NEPAL")
user_question = st.text_input("🇳🇵 🔍Ask a question from the PDF")

if user_question:
    with st.spinner("Generating response..."):

        # Build retriever
        vectordb = build_or_load_vectorstore()
        retriever = vectordb.as_retriever()

        # Prompt template (same as in IPYNB)
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        # Model: use flash version
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2
        )

        # RAG chain
        doc_chain = create_stuff_documents_chain(llm=model, prompt=prompt)
        rag_chain = create_retrieval_chain(retriever, doc_chain)

        # Ask
        response = rag_chain.invoke({"input": user_question})
        st.subheader("📤 Answer")
        st.write(response["answer"])