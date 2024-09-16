import openai
import json
import os
from dotenv import load_dotenv

load_dotenv()

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
models_path = os.path.join(base_dir, 'models.json')

with open(models_path) as f:
    models_metadata = json.load(f)

openai.api_key = os.getenv("OPENAI_API_KEY")

def llm_model_selection(user_message):
    models_metadata_str = json.dumps(models_metadata)
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", 
             "content": 
            f"""
             You are a helpful assistant. You have the following models metadata: {models_metadata_str}.
                If user asks anything, search the model and tell which model_path to use.
                Answer just the modal path no extra information. Just give the model path.
                If the user query links to any of the model descriptions, or tags even the slightest,
                and the user does not ask for a specific model, just give the model path.
                If it's a random question, just answer as you do.
            """
            },
            {"role": "user", "content": user_message}
        ]
    )
    
    return response.choices[0].message.content
    

if __name__ == "__main__":
    user_query = input("User: ")
    selected_model = llm_model_selection(user_query)
    print(f"Selected Model: {selected_model}")
