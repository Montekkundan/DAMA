# random_selection.py
import json
import os
import random

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
models_path = os.path.join(base_dir, 'models.json')


with open(models_path) as f:
    models_metadata = json.load(f)

def random_model_selection(user_message):
    models = models_metadata['models']
    selected_model = random.choice(models)['model_path']
    return selected_model

if __name__ == "__main__":
    user_query = input("User: ")
    selected_model = random_model_selection(user_query)
    print(f"Selected Model: {selected_model}")
