import os
from chat.test import get_completion
import pytest
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, 'test_cases.json')


with open(json_path, 'r') as file:
    test_cases = json.load(file)

test_cases_param = [(case['text'], case['answer']) for case in test_cases]

@pytest.mark.parametrize("query, expected", test_cases_param)
def test_model_selection(query, expected):
    result = get_completion(query).strip()
    
    print(f"Query: '{query}' | Expected: '{expected}' | Got: '{result}'", end=' ')
    
    if expected in result:
        print("✅") 
    else:
        print("❌")
    
    assert expected in result

if __name__ == "__main__":
    pytest.main()
