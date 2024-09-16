from chat.llm_based_selection import llm_model_selection
from chat.embedding_based_selection import embedding_model_selection

def hybrid_model_selection(user_message):
    # get LLM's suggestion
    llm_suggestion = llm_model_selection(user_message)
    
    # get embedding-based suggestion
    embedding_suggestion = embedding_model_selection(user_message)
    
    # Combine both suggestions
    if llm_suggestion == embedding_suggestion:
        return llm_suggestion
    else:
        # Decide which one to trust or ask the user for clarification
        # we prioritize the LLM's suggestion
        return llm_suggestion

if __name__ == "__main__":
    user_query = input("User: ")
    selected_model = hybrid_model_selection(user_query)
    print(f"Selected Model: {selected_model}")
