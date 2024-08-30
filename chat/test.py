from transformers import BertTokenizer, BertModel
import torch
import subprocess
import openai
import os
import json
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

# Load the BERT model and tokenizer once globally
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

with open('models.json') as f:
    models_metadata = json.load(f)

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

memory = ConversationBufferMemory(return_messages=True)

def get_bert_embeddings(text, tokenizer, bert_model):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = bert_model(**inputs)
    # Get the mean of the token embeddings (ignoring the batch dimension)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings

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
    extracted_keywords = {}

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", 
             "content": 
            f"""
            You are a helpful assistant that helps users choose models by printing the model path. You have the following models metadata: {models_metadata_str}.
            Your goal is to assist the user in finding the correct model path based on their query.
            Before you reply with the answer to the user's question/problem, store the extracted keywords into this dictionary variable in the form of 'keywords: [key1, key2, ...]': {extracted_keywords}
            Then print out the set of keywords that capsulate the context of the user's problem, these keywords will show how you understand the situation of the user and will be used to select the best model upon. 
            
            You may ask the user more questions to clarify the context or get more information about the problem, update the keywords set while doing that.
            Below is the guide on how to answer based on the situation:
                If the user's query is related to any description or tag in the models metadata but is too vague, give them a list of models' paths that may help with their problems, ask follow-up questions to determine the specific issue and give the best fit model from the list.
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

        # Save the extracted keywords to generate embeddings
        extracted_keywords = {"keywords": output.split()}  # Simplify keyword extraction for demo
        
        keyword_embeddings = []
        for keyword in extracted_keywords.get('keywords', []):
            keyword_embeddings.append(get_bert_embeddings(keyword, tokenizer, bert_model))
        
        model_embeddings = {}
        for model in models_metadata['models']:
            model_description = model['description']
            model_embeddings[model['model_path']] = get_bert_embeddings(model_description, tokenizer, bert_model)
        
        average_keyword_embedding = np.mean(keyword_embeddings, axis=0)
        
        best_model_path = None
        highest_similarity = -1

        for model_path, model_embedding in model_embeddings.items():
            # Ensure the embeddings are 2D before calculating cosine similarity
            similarity = cosine_similarity([average_keyword_embedding], [model_embedding])[0][0]
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_model_path = model_path
        
        if best_model_path:
            result = run_model(best_model_path)
        else:
            result = "No matching model found."

        memory.save_context({"input": message}, {"output": response.choices[0].message.content})
        return result

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
