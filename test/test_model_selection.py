import sys
import os
import json
from tabulate import tabulate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chat.llm_based_selection import llm_model_selection
from chat.embedding_based_selection import embedding_model_selection
from chat.hybrid_selection import hybrid_model_selection

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
test_queries_path = os.path.join(base_dir, 'test_queries.json')

# Load test queries
with open(test_queries_path) as f:
    test_queries = json.load(f)

def truncate_text(text, max_length=30):
    if len(text) > max_length:
        return text[:max_length-3] + '...'
    return text

def test_selection_methods():
    results = []
    correct_llm = 0
    correct_embedding = 0
    correct_hybrid = 0
    total = len(test_queries)
    
    for test_case in test_queries:
        user_query = truncate_text(test_case['user_query'])
        expected_model = truncate_text(test_case['expected_model'])
        
        llm_model = truncate_text(llm_model_selection(test_case['user_query']))
        embedding_model = truncate_text(embedding_model_selection(test_case['user_query']))
        hybrid_model = truncate_text(hybrid_model_selection(test_case['user_query']))
        
        llm_correct = llm_model == expected_model
        embedding_correct = embedding_model == expected_model
        hybrid_correct = hybrid_model == expected_model
        
        if llm_correct:
            correct_llm += 1
        if embedding_correct:
            correct_embedding += 1
        if hybrid_correct:
            correct_hybrid += 1
        
        results.append([
            user_query, 
            expected_model, 
            llm_model, "✔" if llm_correct else "✘", 
            embedding_model, "✔" if embedding_correct else "✘", 
            hybrid_model, "✔" if hybrid_correct else "✘"
        ])
    
    llm_accuracy = correct_llm / total * 100
    embedding_accuracy = correct_embedding / total * 100
    hybrid_accuracy = correct_hybrid / total * 100

    headers = ["User Query", "Expected Model", 
               "LLM-Based Model", "LLM Correct", 
               "Embedding-Based Model", "Embedding Correct", 
               "Hybrid Model", "Hybrid Correct"]
    
    print(tabulate(results, headers=headers, tablefmt="grid"))
    print(f"LLM-Based Selection Accuracy: {llm_accuracy:.2f}%")
    print(f"Embedding-Based Selection Accuracy: {embedding_accuracy:.2f}%")
    print(f"Hybrid Selection Accuracy: {hybrid_accuracy:.2f}%\n")

if __name__ == "__main__":
    test_selection_methods()
