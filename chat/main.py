import openai
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

with open('models.json') as f:
    models_metadata = json.load(f)



# Set up OpenAI client
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def get_completion(message):
    models_metadata_str = json.dumps(models_metadata)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", 
             "content":  f"""
    You are a helpful assistant that helps users choose models by printing the model path. You have the following models metadata: {models_metadata_str}.
    Your goal is to assist the user in finding the correct model path based on their query.
    If the user's query is related to any description or tag in the models metadata but is too vague, ask follow-up questions to determine the specific issue.
    If the user confirms a specific model-related issue (e.g., diabetes), immediately provide the model path without asking any further questions.
    If the user's query matches multiple models, suggest those models to the user and let them pick. If you're not sure, use your best judgment to choose the most relevant model.
    If there is a clear match with a single model and the user confirms it, provide the model path directly without any additional explanation, context, or sentences. Only print the model path.
    If the query is completely unrelated to any model, answer as you normally would.
"""},
            {"role": "user", "content": message}
        ]
    )
    return response.choices[0].message.content    

def main():
    print("Welcome to the terminal chat app. Type your query below:")
    while True:
        user_query = input("> ")
        if user_query.lower() in ["exit", "quit"]:
            break
        
        answer = get_completion(user_query)
        print(f"Assistant: {answer}")

if __name__ == "__main__":
    main()
