import openai
import google.generativeai as genai
from anthropic import Anthropic
import os
import json
import argparse
from dotenv import load_dotenv
from rich.table import Table
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
import time

load_dotenv()

def create_rich_table(results):
    table = Table(title="Model Selection Results", expand=True)
    
    table.add_column("User Query", style="cyan", width=20)
    table.add_column("Expected Model", style="green", width=20)
    table.add_column("Result", style="yellow", width=10)
    table.add_column("Correct", justify="center", style="green", width=5)

    for i, row in enumerate(results):
        table.add_row(
            row['query'], 
            row['expected_model'], 
            row['result'], 
            "✔️" if row['correct'] else "❌"
        )
    
    return table

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
models_path = os.path.join(base_dir, 'models.json')

with open(models_path) as f:
    models_metadata = json.load(f)

openai.api_key = os.getenv("OPENAI_API_KEY")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

models_metadata_str = json.dumps(models_metadata)
system_message = f"""
You are a helpful assistant. You have the following models metadata: {models_metadata_str}.
If the user asks anything, search the model and tell which model_path to use.
Answer just the model path, no extra information. Just give the model path.
If you give the model path, do not give in quotes, or any other format, just the path.
If the user query links to any of the model descriptions or tags, even the slightest, just give the model path.
If it's a random question, just answer as you normally do.
"""

def llm_model_selection(user_message, api_provider='openai'):
    if api_provider == 'openai':
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content.strip()

    elif api_provider == 'gemini':
        model=genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system_message)
        response = model.generate_content(user_message)
        return response.text.strip()

    elif api_provider == 'anthropic':
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=512,
            system=system_message,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        return response.content[0].text.strip()

def run_test_cases(api_provider='openai'):
    test_cases = [
        {
            "query": "When is the optimal season to sow barley in temperate climates?",
            "expected_model": "predict.py"
        },
        {
            "query": "Will it rain heavily in the next 48 hours?",
            "expected_model": "weather_model.py"
        },
        {
            "query": "Forecast the consumer purchasing behavior for the upcoming holiday season.",
            "expected_model": "sales_model.py"
        },
        {
            "query": "Assess the likelihood of glucose regulation issues in this patient.",
            "expected_model": "./models/diabetes_prediction/predict.py"
        },
        {
            "query": "Should I consider investing in emerging tech equities this quarter?",
            "expected_model": "stock_model.py"
        },
        {
            "query": "Evaluate the patient's risk for cardiovascular complications.",
            "expected_model": "./models/heart_disease_prediction/predict.py"
        },
        {
            "query": "What nutrient additions would improve soil fertility for my crops?",
            "expected_model": "fertilizer_recommendation.py"
        },
        {
            "query": "Predict the electricity usage trends during peak summer months.",
            "expected_model": "energy_forecast.py"
        },
        {
            "query": "What are people saying about our new product on social platforms?",
            "expected_model": "social_media_sentiment.py"
        },
        {
            "query": "Anticipate the vehicular density in urban areas during the festival weekend.",
            "expected_model": "traffic_congestion.py"
        },
        {
            "query": "Is there a potential for a viral infection spread in the community?",
            "expected_model": "disease_outbreak.py"
        },
        {
            "query": "Convert the following document into Mandarin Chinese.",
            "expected_model": "language_translation.py"
        }
    ]

    results = []
    correct = 0
    total = len(test_cases)

    start_time = time.time()

    console = Console()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Processing...", total=total)

        for i, test_case in enumerate(test_cases):
            result = llm_model_selection(test_case['query'], api_provider=api_provider)
            is_correct = result.strip() == test_case['expected_model']
            results.append({
                'query': test_case['query'],
                'expected_model': test_case['expected_model'],
                'result': result,
                'correct': is_correct
            })
            if is_correct:
                correct += 1

            progress.advance(task)

    table = create_rich_table(results)
    console.print(table)

    accuracy = (correct / total) * 100
    console.print(f"Accuracy for {api_provider}: {accuracy:.2f}%")

    end_time = time.time()
    elapsed_time = end_time - start_time
    console.print(f"Total time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model selection using different LLMs.')
    parser.add_argument('-g', '--google', action='store_true', help='Use Google Gemini API')
    parser.add_argument('-a', '--anthropic', action='store_true', help='Use Anthropic API')
    args = parser.parse_args()

    if args.google:
        run_test_cases(api_provider='gemini')
    elif args.anthropic:
        run_test_cases(api_provider='anthropic')
    else:
        run_test_cases(api_provider='openai')
