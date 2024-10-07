from main import generate_ans
from langchain_community.llms.ollama import Ollama
from langchain_chroma.vectorstores import Chroma
# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings

# disable TensorFlow's custom oneDNN operations (disable unnecessary errors in terminal)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Initialize LLM and vector database
llm = Ollama(model="mistral")           # llm 
persist_directory = "./db/db_minilm"    # database directory
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vector_database = Chroma(
    collection_name="local-rag",
    persist_directory=persist_directory,
    embedding_function=embedding
)

# Prompt to get validation
EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

# Validation Function
def query_and_validate(question: str, expected_response: str):
    response_text = generate_ans(question, llm=llm, vector_database=vector_database)
    
    # pass to evaluation prompt
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    evaluation_results_str = llm.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(question)
    print(prompt)
    return print_res(evaluation_results_str_cleaned, question)

# Print result
def print_res(eval_res, query):
    if "true" in eval_res:
        print("\033[92m" + f"Test Passed: {query}" + "\033[0m")
        return True
    elif "false" in eval_res:
        print("\033[91m" + f"Test Failed: {query}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )

# List of test cases
test_cases = [
    {
        "query": "What is velocity?", 
        "expected_response": "Velocity is the rate of change of position."
    },
    {
        "query": "Can you explain the difference between velocity and acceleration?", 
        "expected_response": "Velocity is the rate of change of position, while acceleration is the rate of change of velocity."
    },
    {
        "query": "What is Newton's first law?", 
        "expected_response": "Newton's first law states that an object will remain at rest or in uniform motion unless acted upon by a force."
    },
    {
        "query": "What is the formula for gravitational force?", 
        "expected_response": "The formula for gravitational force is F = G * (m1 * m2) / r^2."
    }
]

# Loop through the test cases and validate each
def run_all_tests():
    for i, test_case in enumerate(test_cases):
        query = test_case["query"]
        expected_response = test_case["expected_response"]
        print(f"\nRunning Test {i+1}...")
        assert query_and_validate(query, expected_response)
        
# Run all tests
if __name__ == "__main__":
    run_all_tests()