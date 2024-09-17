import sys
import os
import json
import tkinter as tk
from tkinter import ttk
from rich.table import Table
from rich.console import Console
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from selection.llm_based_selection import llm_model_selection
from selection.embedding_based_selection import embedding_model_selection
from selection.hybrid_selection import hybrid_model_selection

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
test_queries_path = os.path.join(base_dir, 'test_queries.json')

console = Console()

with open(test_queries_path) as f:
    test_queries = json.load(f)

def truncate_text(text, max_length=40):
    if len(text) > max_length:
        return text[:max_length-3] + '...'
    return text

def create_rich_table(results):
    table = Table(title="Model Selection Results", expand=True)
    
    table.add_column("User Query", style="cyan", width=10)
    table.add_column("Expected Model", style="magenta", width=10)
    table.add_column("LLM-Based Model", style="green", width=10)
    table.add_column("LLM Correct", justify="center", style="bold", width=5)
    table.add_column("Embedding-Based Model", style="yellow", width=10)
    table.add_column("Embedding Correct", justify="center", style="bold", width=5)
    table.add_column("Hybrid Model", style="blue", width=10)
    table.add_column("Hybrid Correct", justify="center", style="bold", width=5)
    
    for row in results:
        table.add_row(*row)
    
    return table

def test_selection_methods():
    correct_llm = 0
    correct_embedding = 0
    correct_hybrid = 0
    total = len(test_queries)
    results = []
    
    for test_case in test_queries:
        user_query = truncate_text(test_case['user_query'], max_length=10)
        expected_model = truncate_text(test_case['expected_model'], max_length=10)
        
        llm_model = truncate_text(llm_model_selection(test_case['user_query']), max_length=10)
        embedding_model = truncate_text(embedding_model_selection(test_case['user_query']), max_length=10)
        hybrid_model = truncate_text(hybrid_model_selection(test_case['user_query']), max_length=10)
        
        llm_correct = llm_model == expected_model
        embedding_correct = embedding_model == expected_model
        hybrid_correct = hybrid_model == expected_model
        
        if llm_correct:
            correct_llm += 1
        if embedding_correct:
            correct_embedding += 1
        if hybrid_correct:
            correct_hybrid += 1

        results.append([user_query, expected_model, llm_model, "✔" if llm_correct else "✘", 
                        embedding_model, "✔" if embedding_correct else "✘", 
                        hybrid_model, "✔" if hybrid_correct else "✘"])
    
    llm_accuracy = correct_llm / total * 100
    embedding_accuracy = correct_embedding / total * 100
    hybrid_accuracy = correct_hybrid / total * 100

    return llm_accuracy, embedding_accuracy, hybrid_accuracy, results

def show_graph(frame, llm_accuracy, embedding_accuracy, hybrid_accuracy):
    for widget in frame.winfo_children():
        widget.destroy()

    accuracies = [llm_accuracy, embedding_accuracy, hybrid_accuracy]
    labels = ['LLM Accuracy', 'Embedding Accuracy', 'Hybrid Accuracy']
    
    fig, ax = plt.subplots()
    ax.bar(labels, accuracies, color=['green', 'yellow', 'blue'])
    ax.set_xlabel('Model Type')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Selection Accuracy Comparison')
    
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

def create_tkinter_table(frame, headers, data):

    for widget in frame.winfo_children():
        widget.destroy()

    for i, header in enumerate(headers):
        header_label = tk.Label(frame, text=header, font=("Arial", 10, "bold"), borderwidth=1, relief="solid", padx=10, pady=5)
        header_label.grid(row=0, column=i, sticky="nsew")

    for row_index, row_data in enumerate(data, start=1):
        for col_index, cell in enumerate(row_data):
            cell_label = tk.Label(frame, text=cell, font=("Arial", 10), borderwidth=1, relief="solid", padx=10, pady=5)
            cell_label.grid(row=row_index, column=col_index, sticky="nsew")

    for i in range(len(headers)):
        frame.grid_columnconfigure(i, weight=1)

def create_gui(llm_accuracy, embedding_accuracy, hybrid_accuracy, results):
    root = tk.Tk()
    root.title("Model Selection Results")

    sidebar = tk.Frame(root, width=200, bg='#333', height=500, relief='sunken', borderwidth=2)
    sidebar.pack(expand=False, fill='both', side='left', anchor='nw')

    content_frame = tk.Frame(root, width=500, height=500, bg='white')
    content_frame.pack(expand=True, fill='both', side='right')

    headers = ["User Query", "Expected Model", "LLM-Based Model", "LLM Correct", 
               "Embedding-Based Model", "Embedding Correct", "Hybrid Model", "Hybrid Correct"]

    def show_llm_accuracy():
        create_tkinter_table(content_frame, headers, results)


    def show_graphs():
        show_graph(content_frame, llm_accuracy, embedding_accuracy, hybrid_accuracy)

    ttk.Button(sidebar, text="Accuracy", command=show_llm_accuracy).pack(fill='x', padx=5, pady=10)
    ttk.Button(sidebar, text="Show Graphs", command=show_graphs).pack(fill='x', padx=5, pady=10)

    show_llm_accuracy()

    root.mainloop()

if __name__ == "__main__":
    if '-h' in sys.argv:
        llm_accuracy, embedding_accuracy, hybrid_accuracy, results = test_selection_methods()

        create_gui(llm_accuracy, embedding_accuracy, hybrid_accuracy, results)
    else:
        llm_accuracy, embedding_accuracy, hybrid_accuracy, results = test_selection_methods()

        table = create_rich_table(results)
        console.print(table)

        print(f"LLM-Based Selection Accuracy: {llm_accuracy:.2f}%")
        print(f"Embedding-Based Selection Accuracy: {embedding_accuracy:.2f}%")
        print(f"Hybrid Selection Accuracy: {hybrid_accuracy:.2f}%\n")

