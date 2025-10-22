from backend.core import run_llm
import streamlit as st

st.header("Langchain Document Helper Bot")

prompt = st.text_input("prompt", placeholder="Enter your query here")

if prompt:
    with st.spinner("Generating response...."):
        generated_response = run_llm(prompt)