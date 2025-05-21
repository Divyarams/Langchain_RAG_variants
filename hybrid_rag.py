from dotenv import load_dotenv
import os
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings

loader=PyMuPDFLoader('redshift-gsg.pdf')
docs=loader.load()
chunking=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
chunks=chunking.split_documents(docs)

from langchain.retrievers.bm25 import BM25Retriever

bm25=BM25Retriever.from_documents(chunks)
bm25.k=5

bm25.invoke('How to create a sample amazon redshift cluster')[0].page_content

from langchain_community.retrievers import PineconeHybridSearchRetriever
import os
import pinecone
index_name="hybrid-search-langchain-v2"

pinecone_client=pinecone.init(api_key='pcsk_rEqg6_MbrEerSe5MWiAfe16PENiz8mEv3cVhHZQUQy2Lr6yPFmgz8psr4BzRiYkeyLyYh',
                              environment='us-west1-gcp')
if index_name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=index_name,
        dimension=768,
        metric='cosine',## for sparse 
        pods=1,
        pod_type="p1.x1"
        

    )

index=pinecone.Index(index_name)
