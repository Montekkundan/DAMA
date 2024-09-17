import json
import os
import re

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
models_path = os.path.join(base_dir, 'models.json')

with open(models_path) as f:
    models_metadata = json.load(f)

keyword_model_map = {}

for model in models_metadata['models']:
    keywords = model.get('keywords', [])
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower not in keyword_model_map:
            keyword_model_map[keyword_lower] = model['model_path']
        else:
            pass

def keyword_model_selection(user_message):
    user_message_lower = user_message.lower()
    for keyword, model_path in keyword_model_map.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', user_message_lower):
            return model_path
    return None

if __name__ == "__main__":
    user_query = input("User: ")
    selected_model = keyword_model_selection(user_query)
    if selected_model:
        print(f"Selected Model: {selected_model}")
    else:
        print("No matching model found.")
