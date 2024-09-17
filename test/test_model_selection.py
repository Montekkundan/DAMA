import sys
import os
import json
import time
from rich.table import Table
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from selection.llm_based_selection import llm_model_selection
from selection.embedding_based_selection import embedding_model_selection
from selection.hybrid_selection import hybrid_model_selection
from selection.keyword_based_selection import keyword_model_selection
from selection.random_selection import random_model_selection

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
test_queries_path = os.path.join(base_dir, 'test_queries.json')

console = Console()

with open(test_queries_path) as f:
    test_queries = json.load(f)

def truncate_text(text, max_length=40):
    if len(text) > max_length:
        return text[:max_length-3] + '...'
    return text

def test_selection_methods():
    correct_llm = 0
    correct_embedding = 0
    correct_hybrid = 0
    correct_keyword = 0
    correct_random = 0
    total = len(test_queries)
    results = []
    
    llm_cache = {}
    embedding_cache = {}
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Processing...", total=total)
        for test_case in test_queries:
            user_query_full = test_case['user_query']
            user_query = truncate_text(user_query_full, max_length=10)
            expected_model = truncate_text(test_case['expected_model'], max_length=10)
            
            if user_query_full in llm_cache:
                llm_model_full = llm_cache[user_query_full]
            else:
                llm_model_full = llm_model_selection(user_query_full)
                llm_cache[user_query_full] = llm_model_full
            llm_model = truncate_text(llm_model_full, max_length=10)
            
            if user_query_full in embedding_cache:
                embedding_model_full = embedding_cache[user_query_full]
            else:
                embedding_model_full = embedding_model_selection(user_query_full)
                embedding_cache[user_query_full] = embedding_model_full
            embedding_model = truncate_text(embedding_model_full, max_length=10)
            
            hybrid_model_full = hybrid_model_selection(user_query_full)
            hybrid_model = truncate_text(hybrid_model_full, max_length=10)
            
            keyword_model_full = keyword_model_selection(user_query_full)
            keyword_model = truncate_text(keyword_model_full if keyword_model_full else '', max_length=10)
            
            random_model_full = random_model_selection(user_query_full)
            random_model = truncate_text(random_model_full, max_length=10)
            
            llm_correct = llm_model == expected_model
            embedding_correct = embedding_model == expected_model
            hybrid_correct = hybrid_model == expected_model
            keyword_correct = keyword_model == expected_model
            random_correct = random_model == expected_model
            
            if llm_correct:
                correct_llm += 1
            if embedding_correct:
                correct_embedding += 1
            if hybrid_correct:
                correct_hybrid += 1
            if keyword_correct:
                correct_keyword +=1
            if random_correct:
                correct_random +=1

            results.append([
                user_query, 
                "✔" if llm_correct else "✘",
                "✔" if embedding_correct else "✘",
                "✔" if hybrid_correct else "✘",
                "✔" if keyword_correct else "✘",
                "✔" if random_correct else "✘"
            ])
            
            progress.advance(task)
    
    llm_accuracy = correct_llm / total * 100
    embedding_accuracy = correct_embedding / total * 100
    hybrid_accuracy = correct_hybrid / total * 100
    keyword_accuracy = correct_keyword / total * 100
    random_accuracy = correct_random / total * 100

    return llm_accuracy, embedding_accuracy, hybrid_accuracy, keyword_accuracy, random_accuracy, results

def create_rich_table(results):
    table = Table(title="Model Selection Results", expand=True)
    
    table.add_column("User Query", style="cyan", width=15)
    table.add_column("LLM Correct", justify="center", style="green", width=10)
    table.add_column("Embedding Correct", justify="center", style="yellow", width=15)
    table.add_column("Hybrid Correct", justify="center", style="blue", width=10)
    table.add_column("Keyword Correct", justify="center", style="magenta", width=15)
    table.add_column("Random Correct", justify="center", style="red", width=15)
    
    for row in results:
        table.add_row(*row)
    
    return table

if __name__ == "__main__":
    start_time = time.time()
    
    llm_accuracy, embedding_accuracy, hybrid_accuracy, keyword_accuracy, random_accuracy, results = test_selection_methods()
    
    table = create_rich_table(results)
    console.print(table)
    
    print(f"LLM-Based Selection Accuracy: {llm_accuracy:.2f}%")
    print(f"Embedding-Based Selection Accuracy: {embedding_accuracy:.2f}%")
    print(f"Hybrid Selection Accuracy: {hybrid_accuracy:.2f}%")
    print(f"Keyword-Based Selection Accuracy: {keyword_accuracy:.2f}%")
    print(f"Random Selection Accuracy: {random_accuracy:.2f}%\n")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")
