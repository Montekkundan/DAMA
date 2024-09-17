import matplotlib.pyplot as plt
import numpy as np

methods = [
    'LLM-Based\nSelection',
    'Embedding-Based\nSelection',
    'Hybrid\nSelection',
    'Keyword-Based\nSelection',
    'Random\nSelection'
]

accuracies = [95.00, 80.00, 95.83, 61.67, 10.83]

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, accuracies, color=['green', 'orange', 'blue', 'purple', 'red'])

for bar, accuracy in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{accuracy:.2f}%', ha='center', va='bottom', fontsize=10)

plt.title('Comparison of Model Selection Accuracy Across Methods', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=12)

plt.ylim(0, 110)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300)
plt.show()
