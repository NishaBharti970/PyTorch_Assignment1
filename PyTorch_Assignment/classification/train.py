import torch
import torch.nn as nn
from model import ClassificationModel
from data_loader import load_data

# Load data
X_train, X_test, y_train, y_test = load_data()

# Model
model = ClassificationModel()

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
epochs = 100

for epoch in range(epochs):
    model.train()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("Training done!")

# 🔥 SAVE MODEL
torch.save(model.state_dict(), "model.pth")
print("Model saved!")