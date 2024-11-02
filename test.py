from main import generate_ans
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from Levenshtein import distance as levenshtein_distance

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
def cosine_similarity(expected, actual, threshold=0.75):
    expected_embedding = model.encode(expected, convert_to_tensor=True)
    actual_embedding = model.encode(actual, convert_to_tensor=True)
    similarity = util.cos_sim(expected_embedding, actual_embedding).item()
    return similarity >= threshold

def jaccard_similarity(expected, actual, threshold=0.5):
    expected_set = set(expected.lower().split())
    actual_set = set(actual.lower().split())
    intersection = expected_set.intersection(actual_set)
    union = expected_set.union(actual_set)
    similarity = len(intersection) / len(union)
    return similarity >= threshold

def levenshtein_similarity(expected, actual, max_distance=10):
    return levenshtein_distance(expected, actual) <= max_distance

def bleu_similarity(expected, actual, threshold=0.5):
    # Apply smoothing function to handle cases with few n-gram overlaps
    smoothie = SmoothingFunction().method1
    score = sentence_bleu([expected.split()], actual.split(), smoothing_function=smoothie)
    return score >= threshold

def rouge_similarity(expected, actual, threshold=0.5):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(expected, actual)['rougeL'].fmeasure
    return score >= threshold

# model for llm validation
llm_model="llama3.2"
def llm_validation(expected, actual):
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
    print("llm_validation_response:",responsed_cleaned)
    
    if "true" in responsed_cleaned:
        return True
    elif "false" in responsed_cleaned:
        return False
    
    return False


# Run test cases and evaluate
def run_tests():
    testcases = load_testcases()
    total_tests = len(testcases)
    print(total_tests, "testcases loaded\n")
    
    results = {
        "total": total_tests,
        "cosine_similarity": 0,
        "jaccard_similarity": 0,
        "levenshtein_similarity": 0,
        "bleu_similarity": 0,
        "rouge_similarity": 0,
        "llm_validation": 0, 
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0
    }

    # read testcases
    for i, case in enumerate(testcases, 1):
        query = case["query"]
        expected_response  = case["expected_response"]
        isCorrect  = case["isCorrect"]
        index  = case["index"]
        
        print(f"\nRunning testcase {i}:")
        print(f"Query: {query}")
        print(f"Expected Response: {expected_response}")
        print(f"Expected Response Status : {isCorrect}\n")

        # Generate actual response
        actual_response = generate_ans(query, llm=llm, embedding_model=embedding_model)
        print(f"Actual Response: {actual_response}\n")

        # Evaluate with each similarity function
        cosine_pass = cosine_similarity(expected_response, actual_response)
        jaccard_pass = jaccard_similarity(expected_response, actual_response)
        levenshtein_pass = levenshtein_similarity(expected_response, actual_response)
        bleu_pass = bleu_similarity(expected_response, actual_response)
        rouge_pass = rouge_similarity(expected_response, actual_response)
        llm_pass = llm_validation(expected_response, actual_response)
        
        # Determine the overall pass status
        passed = cosine_pass or llm_pass
        # Update TP, TN, FP, FN based on isCorrect and passed
        if isCorrect:  # Expected response is correct
            if passed: results['TP'] += 1  # True Positive
            else: results['FN'] += 1  # False Negative
        else:  # Expected response is incorrect
            if passed: results['FP'] += 1  # False Positive
            else: results['TN'] += 1  # True Negative
            
        # Store pass status for each measure considering isCorrect
        cosine_result = cosine_pass == isCorrect
        jaccard_result = jaccard_pass == isCorrect
        levenshtein_result = levenshtein_pass == isCorrect
        bleu_result = bleu_pass == isCorrect
        rouge_result = rouge_pass == isCorrect
        llm_result = llm_pass == isCorrect
        
        # Update results count
        if cosine_result: results["cosine_similarity"] += 1
        if jaccard_result: results["jaccard_similarity"] += 1
        if levenshtein_result: results["levenshtein_similarity"] += 1
        if bleu_result: results["bleu_similarity"] += 1
        if rouge_result: results["rouge_similarity"] += 1
        if llm_result: results["llm_validation"] += 1

        update_result(total_tests,results,i,query,expected_response,actual_response,isCorrect,cosine_result,jaccard_result,levenshtein_result,bleu_result,rouge_pass,llm_result)


def update_result(total_tests,results,i,query,expected_response,actual_response,isCorrect,cosine_result,jaccard_result,levenshtein_result,bleu_result,rouge_result,llm_result):
        # Print individual results
        print("\nResult:")
        print(f"Cosine Similarity = {'passed' if cosine_result else 'failed'}")
        print(f"Jaccard Similarity = {'passed' if jaccard_result else 'failed'}")
        print(f"Levenshtein Similarity = {'passed' if levenshtein_result else 'failed'}")
        print(f"BLEU Similarity = {'passed' if bleu_result else 'failed'}")
        print(f"ROUGE Similarity = {'passed' if rouge_result else 'failed'}")
        print(f"LLM Validation = {'passed' if llm_result else 'failed'}")

        # Append the current test case details to `test_result.txt`
        with open("test_results/test_output.txt", "a", encoding="utf-8") as result_file:
            result_file.write(f"Testcase {i}:\n")
            result_file.write(f"Query: {query}\n")
            result_file.write(f"Expected Response: {expected_response}\n")
            result_file.write(f"Expected Response Status : {isCorrect}\n\n")
            result_file.write(f"Actual Response: {actual_response}\n")

            result_file.write(f"\nResult:\n")
            result_file.write(f"Cosine Similarity: {'passed' if cosine_result else 'failed'}\n")
            result_file.write(f"Jaccard Similarity: {'passed' if jaccard_result else 'failed'}\n")
            result_file.write(f"Levenshtein Similarity: {'passed' if levenshtein_result else 'failed'}\n")
            result_file.write(f"BLEU Similarity: {'passed' if bleu_result else 'failed'}\n")
            result_file.write(f"ROUGE Similarity: {'passed' if rouge_result else 'failed'}\n")
            result_file.write(f"LLM Validation: {'passed' if llm_result else 'failed'}\n\n\n")

        # Write cumulative results to `summary_result.txt` after each iteration
        with open("test_results/test_summary.txt", "w", encoding="utf-8") as summary_file:
            summary_file.write(f"Total Questions = {total_tests}\n\n")
            summary_file.write(f"Total Questions Tested = {i}\n\n")
            summary_file.write(f"Passed Tests:\n")
            
            summary_file.write(f"Cosine Similarity = {results['cosine_similarity']}\n")
            summary_file.write(f"Jaccard Similarity = {results['jaccard_similarity']}\n")
            summary_file.write(f"Levenshtein Similarity = {results['levenshtein_similarity']}\n")
            summary_file.write(f"BLEU Similarity = {results['bleu_similarity']}\n")
            summary_file.write(f"ROUGE Similarity = {results['rouge_similarity']}\n")
            summary_file.write(f"LLM Validation = {results['llm_validation']}\n\n")
            
            # Scores
            TP = results['TP']
            TN = results['TN']
            FP = results['FP']
            FN = results['FN']

            total_passed = results['cosine_similarity'] or results['llm_validation']
            accuracy = total_passed/i*100
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

            summary_file.write(f"(cosine_similarity|llm validation)\n")
            summary_file.write(f"True Positives (TP) = {TP}\n")
            summary_file.write(f"True Negatives (TN) = {TN}\n")
            summary_file.write(f"False Positives (FP) = {FP}\n")
            summary_file.write(f"False Negatives (FN) = {FN}\n\n")
            
            summary_file.write(f"Accuracy = {accuracy:.3f}%\n")
            summary_file.write(f"Precision = {precision:.3f}\n")
            summary_file.write(f"Recall = {recall:.3f}\n")
            summary_file.write(f"F1 Score = {f1_score:.3f}\n")


# Run the test suite
if __name__ == "__main__":
    run_tests()
