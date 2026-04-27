# PyTorch Implementation for Regression and Classification

## 📌 Overview
This project demonstrates PyTorch implementation for:
- Regression Task (Student Score Prediction)
- Classification Task (Iris Dataset)

---

## 📊 Regression Task

### Dataset
- Student Scores Dataset
- Input: Hours studied
- Output: Scores

### Preprocessing
- Data loaded using pandas
- Features normalized using StandardScaler
- Train-test split applied

### Model Architecture
- Input layer: 1 neuron
- Hidden layer: 10 neurons with ReLU
- Output layer: 1 neuron

### Loss Function
- Mean Squared Error (MSE)

### Result
- Model successfully learned relationship between hours and scores
- Loss decreased during training

---

## 🎯 Classification Task

### Dataset
- Iris Dataset (from sklearn)
- Input: 4 features
- Output: 3 classes

### Preprocessing
- StandardScaler used for normalization
- Train-test split applied

### Model Architecture
- Input layer: 4 neurons
- Hidden layer: 10 neurons with ReLU
- Output layer: 3 neurons

### Loss Function
- CrossEntropyLoss

### Result
- Model achieved high accuracy (~90–100%)
- Successfully classified different flower species

---

## ⚙️ How to Run

### Install dependencies
pip install -r requirements.txt

---

### Run Regression
cd regression
python train.py
python test.py

---

### Run Classification
cd classification
python train.py
python test.py

---

## 📁 Project Structure
- regression/
- classification/
- README.md
- requirements.txt

---

## 📌 Conclusion
Both regression and classification models were implemented successfully using PyTorch. The models performed well on their respective tasks and demonstrated effective learning.