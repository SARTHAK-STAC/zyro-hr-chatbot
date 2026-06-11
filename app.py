import streamlit as st

st.title("Zyro HR Chatbot")

question = st.text_input("Ask a question")

if question:
    st.write("You asked:", question)
