from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings 
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
import os


load_dotenv()

def run_llm(query: str):
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        azure_deployment="text-embedding-ada-002",
    )

    vector_store = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings
    )

    retrieval_qa_chat_prompt: PromptTemplate = hub.pull("langchain-ai/retrieval-qa-chat")

    llm = AzureChatOpenAI(
        model="gpt-4o",
        temperature=0,
    )

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=combine_docs_chain,
    )

    result = retrival_chain.invoke({"input": query})
    new_result = {
        "query": result['input'],
        "result": result['answer'],
        "source_documents": result['context']
    }
    return new_result

if __name__ == "__main__":
    res = run_llm(query="What is a LangChain Chain?")
    print(res['result'])
