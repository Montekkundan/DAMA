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
             "content": f"""
                You are a helpful assistant. You have the following models metadata: {models_metadata_str}.
                If user asks anything, search the model and tell which model_path to use.
                Answer just the modal path no extra information. Just give the model path.
                If the user query links to any of the model descriptions, or tags even the slightest,
                and the user does not ask for a specific model, just give the model path.
                If it's a random question, just answer as you do.
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
