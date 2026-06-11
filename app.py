import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

st.set_page_config(page_title="Zyro HR Chatbot")
st.title("🤖 Zyro HR Chatbot")

# Get Groq API key from Streamlit Secrets
groq_key = st.secrets["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = groq_key


@st.cache_resource
def load_rag():

    loader = PyPDFDirectoryLoader(".")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    return retriever, llm


retriever, llm = load_rag()

question = st.text_input("Ask an HR question")

if question:

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Answer ONLY using the context below.

If the answer is not present in the context, reply exactly:

I can only answer questions based on Zyro Dynamics HR policy documents.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    st.write(response.content)
