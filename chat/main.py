import subprocess
import openai
import os
import json
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

load_dotenv()

with open('models.json') as f:
    models_metadata = json.load(f)

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

memory = ConversationBufferMemory(return_messages=True)

def run_model(model_path):
    """Executes the selected model by running the script at the given path.

    Args:
        model_path (str): The path to the Python script that should be executed.

    Returns:
        str: The standard output of the script if execution is successful, 
        or an error message if the script fails.
    """
    result = subprocess.run(['python', model_path], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()  # Return the output if successful
    else:
        return f"Error running the model at {model_path}: {result.stderr}"


function_definitions = [
    {
        "name": "run_model",
        "description": "Run the selected model based on the provided model path.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_path": {
                    "type": "string",
                    "description": "The path to the model that should be run."
                }
            },
            "required": ["model_path"]
        }
    }
]

def get_completion(message):
    models_metadata_str = json.dumps(models_metadata)
    history = memory.load_memory_variables({})['history']

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", 
             "content": 
            f"""
            You are a helpful assistant that helps users choose models by printing the model path. You have the following models metadata: {models_metadata_str}.
            Your goal is to assist the user in finding the correct model path based on their query.
            If the user's query is related to any description or tag in the models metadata but is too vague, ask follow-up questions to determine the specific issue.
            If the user confirms a specific model-related issue (e.g., diabetes), immediately provide the model path without asking any further questions.
            If the user's query matches multiple models, suggest those models to the user and let them pick. If there is a clear match with a single model and the user confirms it, provide the model path directly.
            If the query is unrelated to any model, answer as you normally would.
            Current conversation history: {history}
            """
            },
            {"role": "user", "content": message}
        ],
        functions=function_definitions
    )
    
    if response.choices[0].message.function_call:
        tool_call = response.choices[0].message.function_call
        if tool_call.name == 'run_model':
            model_path = json.loads(tool_call.arguments)['model_path']
            result = run_model(model_path)
            return result
    else:
        output = response.choices[0].message.content.strip()
        if output.startswith('./'):
            result = run_model(output)
            return result
        memory.save_context({"input": message}, {"output": response.choices[0].message.content})
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