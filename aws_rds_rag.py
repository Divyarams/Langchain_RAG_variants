from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import DistanceStrategy, PGVector
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()


llm=ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'],model='llama3-8b-8192')

loader=PyPDFLoader('redshift-gsg.pdf')
docs=loader.load()
chunking=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=chunking.split_documents(docs)
connection_string = f"postgresql://postgres:PWD@ENDPOINT:PORT/DATABASE"
db = PGVector(
    connection_string=connection_string,
    collection_name='mydivya',
    embedding_function=OllamaEmbeddings(model='nomic-embed-text'),
    distance_strategy=DistanceStrategy.COSINE
    
)

##Load from texts
db.from_texts(
                    texts=datas,
                    collection_name=COLLECTION1,
                    connection_string=connection_string,
                    embedding=OllamaEmbeddings(model='nomic-embed-text')
                )

db.from_documents(
                    documents=chunks,
                    collection_name=COLLECTION2,
                    connection_string=connection_string,
                    
                    embedding=OllamaEmbeddings(model='nomic-embed-text')
                )

chain_rag=RetrievalQA.from_chain_type(llm=llm,
                                   retriever=db.as_retriever(search_kwargs={'k':4},
                                                             verbpse=True))


### To delete from db for any rag updates - use the following
db.delete(
    ids=None,  # Set to specific IDs if you want to delete particular documents
    delete_all=False,  # Set to True to delete everything in collection
    filter={"title": "Amazon Redshift-Getting Started Guide"}  # Metadata filter
)
