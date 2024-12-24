from main import generate_ans
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer, util
from bert_score import score
# from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRelevancyMetric, HallucinationMetric
# from deepeval.test_case import LLMTestCase
import pandas as pd

# set test parameters (llm and database)
llm = "mistral"
embedding_model = "all-MiniLM-L6-v2"

# Load test cases
def load_testcases(file_path = "test_results/testcases.txt"):
    testcases = []
    case_data = {}  # Temporarily store each case's data
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                
                # Start of a new test case
                if(line.startswith("query:")):
                    # Save the previous test case if it exists
                    if case_data: testcases.append(case_data)
                    
                    # Start a new test case dictionary    
                    case_data = {"query": line.split("query:", 1)[1].strip()}
                
                elif line.startswith("expected_response:"):
                    case_data["expected_response"] = line.split("expected_response:", 1)[1].strip()

                elif line.startswith("isCorrect :"):
                    case_data["isCorrect"] = line.split("isCorrect :", 1)[1].strip().lower() == "true"

            if case_data: testcases.append(case_data)
        
        # Add index to each test case
        for index, case in enumerate(testcases, start=1):
            case["index"] = index
                
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        exit()
    except UnicodeDecodeError:
        print(f"Could not decode the file {file_path}. Please check the file encoding.")
        exit()

    return testcases


# Load sentence transformer model for cosine similarity
model = SentenceTransformer('all-MiniLM-L6-v2')
def cosine_similarity(expected, actual):
    expected_embedding = model.encode(expected, convert_to_tensor=True)
    actual_embedding = model.encode(actual, convert_to_tensor=True)
    similarity = util.cos_sim(expected_embedding, actual_embedding).item()
    return similarity

def bert_score_similarity(expected, actual, model="bert-base-uncased"):
    expected = [expected]
    actual = [actual]
    P, R, F1 = score(actual, expected, model_type=model, verbose=False) # Compute BERTScore
    return float(F1[0]) # Return F1 score (primary metric for semantic similarity)

# model for llm validation
llm_model="llama3.2"
def llm_validation(expected, actual):
    try:
        llm = OllamaLLM(model=llm_model)
        
        # evaluation prompt template
        EVAL_PROMPT = """
        Expected Response: {expected_response}
        Actual Response: {actual_response}
        ---
        Determine if the actual response captures the key idea or information from the expected response. just check if they both at least mean the same thing, doesn't have to match all the details. Provide only 'true' or 'false' as the answer (in single word).
        """
        prompt = EVAL_PROMPT.format(
            expected_response=expected,
            actual_response=actual
        )
        response = llm.invoke(prompt)
        responsed_cleaned = response.lower().strip()
        return "true" in responsed_cleaned
    
    except Exception as e:
        print(f"Error in LLM validation: {e}")
        return False

# # DeepEval Metrics
# deepeval_metrics = {
#     "AnswerRelevancyMetric": AnswerRelevancyMetric(threshold=0.7, model="mistral", include_reason=True),
#     "FaithfulnessMetric": FaithfulnessMetric(threshold=0.7, model="mistral", include_reason=True),
#     "ContextualRelevancyMetric": ContextualRelevancyMetric(threshold=0.7, model="mistral", include_reason=True),
#     # "HallucinationMetric": HallucinationMetric(threshold=0.7, model="mistral", include_reason=True),
# }


# Calculate TP, TN, FP, FN, and metrics for each method
def calculate_metrics(testcases, results, method):
    TP = results[f'{method}_TP']
    TN = results[f'{method}_TN']
    FP = results[f'{method}_FP']
    FN = results[f'{method}_FN']
    
    accuracy = (TP + TN) / len(testcases) if len(testcases) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

# Update the results dictionary
def update_metrics(results, method, isCorrect, passed):
    if isCorrect:
        if passed:
            results[f"{method}_TP"] += 1
        else:
            results[f"{method}_FN"] += 1
    else:
        if passed:
            results[f"{method}_FP"] += 1
        else:
            results[f"{method}_TN"] += 1
    return results

# Run test cases and evaluate
def run_tests():
    testcases = load_testcases()
    total_tests = len(testcases)
    print(total_tests, "testcases loaded\n")
    
    log_file = "test_results/test_log.txt"
    open(log_file, "w").close()  # Clear the log file before starting
    
    # Initialize results
    results = {
        "cosine_similarity_TP": 0, "cosine_similarity_TN": 0,
        "cosine_similarity_FP": 0, "cosine_similarity_FN": 0,
        "bert_score_TP": 0, "bert_score_TN": 0,
        "bert_score_FP": 0, "bert_score_FN": 0,
        "llm_validation_TP": 0, "llm_validation_TN": 0,
        "llm_validation_FP": 0, "llm_validation_FN": 0,
    }
    
    # Create a DataFrame to store metrics
    results_df = pd.DataFrame(columns=["Evaluation_Metric", "Case_Tested", "Passed_Tests", "Accuracy", "Precision", "Recall", "F1_Score"])
    results_df["Evaluation_Metric"] = ["Cosine Similarity", "BERT Score Similarity", "LLM Validation"]
    results_df.set_index("Evaluation_Metric", inplace=True)

    # Initialize values in the DataFrame
    results_df["Case_Tested"] = 0
    results_df["Passed_Tests"] = 0
    results_df["Accuracy"] = 0.0
    results_df["Precision"] = 0.0
    results_df["Recall"] = 0.0
    results_df["F1_Score"] = 0.0
    
    # read testcases
    for i, case in enumerate(testcases, 1):
        query = case["query"]
        expected_response  = case["expected_response"]
        isCorrect  = case["isCorrect"]
        index  = case["index"]
        
        print(f"\nRunning testcase {i}:\nQuery: {query}\nExpected Response: {expected_response}\n")

        # Generate actual response
        actual_response, retrieved_docs = generate_ans(query, llm=llm, embedding_model=embedding_model, chat_history=[])
        print(f"\nActual Response: {actual_response}\n")

        # Cosine similarity evaluation
        cosine_score = cosine_similarity(expected_response, actual_response)
        cosine_pass = cosine_score >= 0.7
        results = update_metrics(results, "cosine_similarity", isCorrect, cosine_pass)

        # BERT score evaluation
        bert_score = bert_score_similarity(expected_response, actual_response)
        bert_pass = bert_score >= 0.7
        results = update_metrics(results, "bert_score", isCorrect, bert_pass)

        # LLM validation
        llm_pass = llm_validation(expected_response, actual_response)
        results = update_metrics(results, "llm_validation", isCorrect, llm_pass)
        print(f"\nExpected Response Status: {isCorrect}\nCosine Output\t\t: {cosine_pass}\nBERT Output\t\t: {bert_pass}\nLLM Output\t\t: {llm_pass}\n")
        
        update_log_file(log_file, case, actual_response, cosine_pass, cosine_score, bert_pass, bert_score, llm_pass)
        
        # Update the DataFrame for each method
        for method in ["cosine_similarity", "bert_score", "llm_validation"]:
            # Use the full metric name here for display
            if method == "cosine_similarity": metric_name = "Cosine Similarity"
            elif method == "bert_score": metric_name = "BERT Score Similarity"
            elif method == "llm_validation": metric_name = "LLM Validation"
            
            metrics = calculate_metrics(testcases[:i], results, method)  # Calculate metrics only for processed cases
            
            # Update DataFrame
            results_df.at[metric_name, "Case_Tested"] = i
            results_df.at[metric_name, "Passed_Tests"] = results[f"{method}_TP"] + results[f"{method}_TN"]
            results_df.at[metric_name, "Accuracy"] = metrics["accuracy"]
            results_df.at[metric_name, "Precision"] = metrics["precision"]
            results_df.at[metric_name, "Recall"] = metrics["recall"]
            results_df.at[metric_name, "F1_Score"] = metrics["f1_score"]

        results_df.to_csv("test_results/test_summary.csv", index=True)
        
        # # DeepEval
        # deepeval_results_df = pd.DataFrame(columns=["Case_no", "Metric", "Score","Reason",])
        # test_case_deepeval = LLMTestCase(
        #     input=query,
        #     actual_output=actual_response,
        #     expected_output=expected_response,
        #     retrieval_context= [doc.page_content for doc in retrieved_docs],
        # )
        # for metric_name, metric in deepeval_metrics.items():
        #     metric.measure(test_case_deepeval)
        #     print("DeepEval Metrics:")
        #     print(f"{metric_name} Score: {metric.score:.4f}")
        #     print(f"{metric_name} Reason: {metric.reason}")
    
        #     # Append the result for the current metric
        #     deepeval_results_df = pd.concat([
        #         deepeval_results_df,
        #         pd.DataFrame({
        #             "Case_no": [i],
        #             "Metric": [metric_name],
        #             "Score": [round(metric.score, 4)],
        #             "Reason": [metric.reason]
        #         })
        #     ], ignore_index=True)
        #     deepeval_results_df.to_csv("test_results/deep_eval_summary.csv", index=False)

    
    # Print final results df
    print("\nResults DataFrame:")
    print(results_df, "\n")

# Create and update log file
def update_log_file(log_file, testcase, actual_response, cosine_pass, cosine_score, bert_pass, bert_score, llm_pass):
    log_content = f"Testcase {testcase['index']}:\n"
    log_content += f"Query: {testcase['query']}\n"
    log_content += f"Expected Response: {testcase['expected_response']}\n"
    log_content += f"\nActual Response: {actual_response}\n"
    
    log_content += f"\nExpected Response Status: {'true' if testcase['isCorrect'] else 'false'}"
    log_content += f"\nCosine Output\t\t: {cosine_pass}\nBERT Output\t\t: {bert_pass}\nLLM Output\t\t: {llm_pass}\n"
    
    log_content += f"\nResult:\n"
    log_content += f"Cosine Similarity\t: {'passed' if cosine_pass else 'failed'} (Score: {cosine_score:.4f})\n"
    log_content += f"BERT Score Similarity\t: {'passed' if bert_pass else 'failed'} (Score: {bert_score:.4f})\n"
    log_content += f"LLM Validation\t\t: {'passed' if llm_pass else 'failed'}\n\n"

    print(f"Result:\nCosine Similarity\t: {'passed' if cosine_pass else 'failed'} (Score: {cosine_score:.4f})\nBERT Score Similarity\t: {'passed' if bert_pass else 'failed'} (Score: {bert_score:.4f})\nLLM Validation\t\t: {'passed' if llm_pass else 'failed'}\n\n")
    
    # Write to log file
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(log_content)


# Run the test suite
if __name__ == "__main__":
    run_tests()
