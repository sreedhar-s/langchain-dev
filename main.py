from backend.core import run_llm
import streamlit as st

st.header("Langchain Document Helper Bot")

prompt = st.text_input("prompt", placeholder="Enter your query here")

def create_sources_string(source_urls: set[str]) -> str:
    if not source_urls:
        return "No source documents found."
    
    sources_list = list(source_urls)
    sources_list.sort()

    source_string = "Sources:\n\n"
    for i, source in enumerate(sources_list):
        source_string += f"{i+1}. {source}\n"
    return source_string


if prompt:
    with st.spinner("Generating response...."):
        generated_response = run_llm(prompt)
        sources = set(doc.metadata['source'] for doc in generated_response['source_documents'])

        formatted_response = {
            f"{generated_response['result']} \n\n {create_sources_string(sources)}"
        }

        st.write(formatted_response)
