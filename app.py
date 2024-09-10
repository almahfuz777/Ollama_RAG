import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import logging


# Load the persisted ChromaDB vector store
embedding = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
persist_directory = "./db"

vector_database = Chroma(
    collection_name="local-rag",
    persist_directory=persist_directory,
    embedding_function=embedding
)

# Initialize the Ollama LLaMA 3.1 model
llm = Ollama(model="llama3.1")


QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model specialized in physics. Your task is to reformulate the following question into five different versions to retrieve the most relevant physics documents.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_database.as_retriever(search_kwargs={"k": 3}),
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt
template = """You are a helpful assistant trained to answer physics questions based on the provided context.
Use only the context below to answer the following question as clearly as possible:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

def post_process_answer(answer):
    # Ensure the answer is focused on the physics topic
    # Optionally trim any irrelevant parts or hallucinated information
    processed_ans =  answer.split("Answer:")[-1].strip()
    return processed_ans if processed_ans else "No relevant info found in the context..."

# Streamlit UI
st.title("Q/A bot Physics")

user_ques = st.text_input("Throw your problems.")
if st.button("Submit"):
    if user_ques:
        try:
            with st.spinner("Fetching AI response..."):
                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
            answer = chain.invoke(user_ques)
            final_ans = post_process_answer(answer)
            st.write(f"Answer: {final_ans}")
        except Exception as e:
            st.error(f"An error occured: {str(e)}")
    else:
        st.write("Please enter a question!")


# logging after each response generation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
def log_query(user_question, answer):
    logger.info(f"Question: {user_question}, Answer: {answer}")