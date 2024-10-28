import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
    You are an AI language model specialized in physics. Your task is to reformulate the following question into five different versions to retrieve the most relevant physics documents.
    Original question: {question}
    """,
)

# RAG prompt
template = """You are a helpful assistant trained to answer physics questions based on the provided context.
Use only the context below to answer the following question as clearly as possible:
{context}
Question: 
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

def post_process_answer(answer):
    # Ensure the answer is focused on the physics topic
    # Optionally trim any irrelevant parts or hallucinated information
    processed_ans =  answer.split("Answer:")[-1].strip()
    return processed_ans if processed_ans else "No relevant info found in the context..."

def generate_ans(question, llm, embedding_model):
    vector_database = load_db(embedding_model)
    llm = OllamaLLM(model=llm)

    # Dynamic retriever that uses the passed LLM
    retriever = MultiQueryRetriever.from_llm(
        vector_database.as_retriever(search_kwargs={"k": 3}),
        llm,
        prompt = QUERY_PROMPT
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke(question)
    final_ans = post_process_answer(answer)
    return final_ans
