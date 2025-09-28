# 🩺 Breast Cancer SVM Classifier

This repository contains a simple machine learning project that uses **Support Vector Machines (SVM)** to classify breast cancer tumors as **malignant** or **benign** using the [Breast Cancer Wisconsin dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) from `scikit-learn`.

---

## 📚 Project Overview

- Dataset: Breast Cancer Wisconsin dataset (loaded via `sklearn.datasets`)
- Model: Support Vector Machine (SVM) with a **linear kernel**
- Preprocessing: Standardization using `StandardScaler`
- Evaluation: Accuracy, Confusion Matrix, and Classification Report

This project demonstrates a simple, yet effective, supervised learning pipeline using Python’s **scikit-learn**.

---

## ⚙️ How It Works

1. **Load Dataset**  
   Loads the breast cancer dataset from `scikit-learn`.

2. **Split Data**  
   Splits the dataset into training and testing sets.

3. **Scale Features**  
   Uses `StandardScaler` to standardize the features.

4. **Train Model**  
   Trains an SVM classifier (linear kernel) to distinguish between malignant and benign tumors.

5. **Evaluate Model**  
   Outputs accuracy, confusion matrix, and classification report.

---

## 📝 Code Example

`python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target  # 0 = malignant, 1 = benign

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM
svm_model = SVC(kernel='linear', C=1, gamma='scale')
svm_model.fit(X_train, y_train)

# Evaluate
y_pred = svm_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=data.target_names))
📊 Example Output
lua
Copy code
Accuracy: 0.9736842105263158
[[39  1]
 [ 2 72]]
              precision    recall  f1-score   support

   malignant       0.95      0.97      0.96        40
      benign       0.99      0.97      0.98        74

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114
📦 Requirements
Python 3.7+

scikit-learn

numpy

Install dependencies:

bash
Copy code
pip install scikit-learn numpy
🚀 Run the Script
Clone the repo and run the script:

bash
Copy code
git clone https://github.com/your-username/breast-cancer-svm-classifier.git
cd breast-cancer-svm-classifier
python breast_cancer_svm.py
📌 Notes
You can experiment with other kernels such as 'rbf' or 'poly' by changing the kernel parameter in SVC.

The dataset is already included with scikit-learn, so no additional downloads are needed.

📝 License
This project is licensed under the MIT License."# breast-cancer-svm-classifier" 
