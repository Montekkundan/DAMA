import pytest
from main import get_completion

# Test cases
test_cases = [
    ("weather app", "weather_model.py"),
    ("farming prediction", "predict.py"),
    ("business forecast", "sales_model.py")
]

@pytest.mark.parametrize("query, expected", test_cases)
def test_model_selection(query, expected):
    result = get_completion(query).strip()
    
    print(f"Query: '{query}' | Expected: '{expected}' | Got: '{result}'", end=' ')
    
    if result == expected:
        print("✅") 
    else:
        print("❌")
    
    assert result == expected

if __name__ == "__main__":
    pytest.main()
