import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import joblib

class Model(nn.Module):
    def __init__(self, input_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_features, 5)
        self.fc2 = nn.Linear(5, 4)
        self.fc3 = nn.Linear(4, 3)
        self.fc4 = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

loaded_net = Model(input_features=7)
loaded_net.load_state_dict(torch.load('diabetes_model.pth', weights_only=True))
loaded_net.eval()

sc = StandardScaler()

X_train = np.array([[6, 148, 72, 35, 0, 33.6, 50],
                    [1, 85, 66, 29, 0, 26.6, 31],
                    [8, 183, 64, 0, 0, 23.3, 32]])
sc.fit(X_train)

joblib.dump(sc, 'scaler.pkl')

sample_data = np.array([[6, 148, 72, 35, 0, 33.6, 50]])
sample_data = sc.transform(sample_data)
sample_data = torch.tensor(sample_data, dtype=torch.float32)

with torch.no_grad():
    prediction = loaded_net(sample_data)

probability = prediction.item()
result = "Positive" if probability > 0.5 else "Negative"

# print(f"Probability of diabetes: {probability:.4f}")
# print(f"Prediction: {result}")
print(result)
