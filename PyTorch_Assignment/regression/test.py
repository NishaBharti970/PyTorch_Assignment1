import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model import RegressionModel

# Load data
data = pd.read_csv("data/student_scores.csv")

X = data[['Hours']].values
y = data[['Scores']].values

# Normalize (same as training)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.FloatTensor(X)

# Load model
model = RegressionModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

with torch.no_grad():
    predictions = model(X)

print("Sample Predictions:")
print(predictions[:5])