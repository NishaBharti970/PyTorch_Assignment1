import torch
from model import ClassificationModel
from data_loader import load_data

# Load data
X_train, X_test, y_train, y_test = load_data()

# Load model
model = ClassificationModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

accuracy = (predicted == y_test).sum().item() / len(y_test)

print(f"Accuracy: {accuracy * 100:.2f}%")