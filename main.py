import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langsmith import Client

load_dotenv()
client = Client()

def main():
    print("Loading document...")
    # loader = TextLoader("D:\Langchain\pdf_chatbot\medium_blog.txt")
    # document = loader.load()
    pdf_path = "D:/Langchain/pdf_chatbot/231210997v5.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    document = loader.load()

    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Number of chunks created: {len(texts)}")

    # Create embeddings
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        azure_deployment="text-embedding-ada-002",
    )

    print("Creating Pinecone vector store...")
    vector_store = PineconeVectorStore.from_documents(
        documents=texts, embedding=embeddings, index_name=os.getenv("INDEX_NAME")
    )
    print("Vector store created successfully.")

    prompt = client.pull_prompt("langchain-ai/retrieval-qa-chat") 

    # Create chat model
    llm = AzureChatOpenAI(
        model="gpt-4o",
        temperature=0,
    )

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=combine_docs_chain,
    )

    query = "Give me a summary of the document."
    result = retrival_chain.invoke({"input": query})

    print(result.keys())


if __name__ == "__main__":
    main()
