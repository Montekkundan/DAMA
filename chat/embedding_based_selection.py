import os
from transformers import BertTokenizer, BertModel
import torch
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
models_path = os.path.join(base_dir, 'models.json')


with open(models_path) as f:
    models_metadata = json.load(f)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

def embedding_model_selection(user_message):
    user_embedding = get_embeddings(user_message)
    best_similarity = -1
    best_model_path = None

    for model in models_metadata['models']:
        model_description = model['description'] + ' ' + ' '.join(model['tags'])
        model_embedding = get_embeddings(model_description)
        similarity = cosine_similarity(user_embedding, model_embedding)[0][0]
        if similarity > best_similarity:
            best_similarity = similarity
            best_model_path = model['model_path']
    
    return best_model_path

if __name__ == "__main__":
    user_query = input("User: ")
    selected_model = embedding_model_selection(user_query)
    print(f"Selected Model: {selected_model}")
