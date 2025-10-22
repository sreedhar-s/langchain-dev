import asyncio
import os
import ssl
from typing import Any, List, Dict

import certifi
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

load_dotenv()

#configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

embeddings = AzureOpenAIEmbeddings(
    model = "text-emdedding-ada-002",
    azure_deployment="text-embedding-ada-002",
    chunk_size = 50,
    retry_min_seconds=10
)

vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings
)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

async def index_documnets_async(documents: List[Document], batch_size: int = 500):
    #process documents in batches asynchronously

    print(f"VectorStore Indexing: starting to index {len(documents)} documents into Pinecone vector store")

    #create batches
    batches = [
        documents[i:i+batch_size] for i in range(0, len(documents), batch_size)
    ]

    print(f"VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each")


    #Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            print(f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)")

        except Exception as e:
            print(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True
    
    #process batches concurrently
    tasks = [
        add_batch(batch, idx+1) for idx, batch in enumerate(batches)
    
    ]

    results = await asyncio.gather(*tasks)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        print("VectorStore Indexing: All batches indexed successfully")
    else:
        print(f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully")

async def main():
    """Min async function to orchestrate entire process"""
    print("Documentation Ingestion Pipeline")

    print("TavilyCrawl: starting to Crawl documentation from https://python.langchain.com")

    #crawl the documentation site
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com",
        "max_depth": 1,
        "extract_depth": "advanced",
         "instructions": "Documentatin relevant to ai agents",
    })

    all_docs = [Document(page_content=doc['raw_content'], metadata={"source": doc['url']}) for doc in res['results']]
    print(f"TavilyCrawl: completed crawling, found {len(all_docs)} documents")

    #Split the documents into chunks
    print("Document chunking phase")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splitted_docs = text_splitter.split_documents(all_docs)
    print(f"Document chunking phase completed, created {len(splitted_docs)} chunks")

    # Process documents asynchronously and add to vector store
    await index_documnets_async(splitted_docs, batch_size=500)
    

if __name__ == "__main__":
    asyncio.run(main())