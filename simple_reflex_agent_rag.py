from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.agents import AgentExecutor,initialize_agent,StructuredChatAgent,AgentType
from langchain.tools import tool
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
import os
llm=ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'],model='llama3-8b-8192',temperature=0.1)

loader=PyPDFLoader('redshift-gsg.pdf')
docs=loader.load()
chunking=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=chunking.split_documents(docs)


@tool
def add_docs(docs):
    """ Adds docs"""
    db=FAISS.add_documents(docs)
    print('Documents added')
    return db
@tool
def query_docs(llm,db,query):
    """ Queries using NLP"""
    print('Query chain')
    chain=RetrievalQA.from_chain_type(llm=llm,
                                      retriever=db.as_retriever(search_kwargs={'k':3}))
    return chain.run(query)


tools=[add_docs,query_docs]
agent = StructuredChatAgent.from_llm_and_tools(
    llm=llm,
    tools=tools,
    prefix=CUSTOM_PROMPT
    # Custom prompt can be added here if needed
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True  # Show reasoning steps
)


general_agent=initialize_agent(
    tools=[],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
# Router agent with routing logic
router_agent = initialize_agent(
    tools=[add_docs,query_docs],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={
        "prefix": """You are a routing expert. Decide which tool to use:
        Use 'add_docs' ONLY when the argument passed is document type
        - Use 'query_docs' ONLY when user sends a query to be executed against the vector store.
        
        
        Return ONLY the tool name (add_docs,query_docs) as your final answer."""
    }
)

def route_query(query: str) -> AgentExecutor:
    """Determine which agent to use"""
    decision = router_agent.run(query)
    
    if "add_docs" in decision:
        return initialize_agent([add_docs], llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)
    elif "query_docs" in decision:
        return initialize_agent([query_docs], llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)
    else:
        return general_agent



