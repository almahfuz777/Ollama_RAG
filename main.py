import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage


def load_db(embedding_model):
    if embedding_model == "nomic-embed-text":
        embedding = OllamaEmbeddings(model="nomic-embed-text")
        persist_directory = "./db/db_nomic"
    elif embedding_model == "all-MiniLM-L6-v2":
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        persist_directory = "./db/db_minilm"
    
    # Load the database with selected embedding model
    vector_database = Chroma(
        collection_name="local-rag",
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    return vector_database

# Query Prompt
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# RAG prompt
template = """You are a friendly assistant trained to answer physics questions for school level student (9th/10th grade). 
Use the conversation history to understand the context to answer the question clearly.

Conversation History:
{conversation_history}

You can use the following context if it's relevant to the question (ignore if it's not relevant):
{context}
Question: 
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

def format_chat_to_str(chat_history):
    """Format the chat history for display in prompts."""
    return "\n".join([f"{entry['role']}: {entry['content']}" for entry in chat_history])

def format_chat_history_for_langchain(chat_history):
    """Format chat history into a list of BaseMessage objects."""
    formatted_history = []
    for entry in chat_history:
        if isinstance(entry, dict):
            role = entry.get("role")
            content = entry.get("content")
            
            if role == "user":
                formatted_history.append(HumanMessage(content=content))
            elif role == "assistant":
                formatted_history.append(AIMessage(content=content))
    return formatted_history

def generate_ans(question, llm, embedding_model, chat_history):
    llm = OllamaLLM(model=llm)
    vector_database = load_db(embedding_model)
    recent_chat = chat_history[-5:]

    # Create a retriever
    retriever = vector_database.as_retriever(search_kwargs={"k": 5})

    # Create the history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, 
        retriever, 
        contextualize_q_prompt
    )
    
    # Retrieve relevant documents
    retrieved_docs = history_aware_retriever.invoke({
        "input": question, 
        "chat_history": format_chat_history_for_langchain(recent_chat)
    })
    # print("retrieved_docs:" ,retrieved_docs)

    chain = (
        RunnableMap({
            "context": history_aware_retriever,
            "question": RunnablePassthrough(),              # This will be the input question
            "conversation_history": RunnablePassthrough()   # This will pass formatted chat history
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    ans = chain.invoke({
        "question": question,
        "conversation_history": format_chat_to_str(recent_chat),  # Pass formatted chat history to the prompt
        "input": question, 
        "chat_history": format_chat_history_for_langchain(recent_chat)  # This is the correctly formatted chat history for the retriever
    })
    
    # ans = ""
    reformulated_question="Question: reformulated_question"
    return ans, retrieved_docs
