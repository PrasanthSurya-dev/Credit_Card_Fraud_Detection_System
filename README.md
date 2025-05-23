# 💳 Credit Card Fraud Detection System

This project uses machine learning to detect fraudulent credit card transactions. The dataset used is highly imbalanced and contains only a small fraction of fraudulent transactions. This system is designed to help financial institutions identify suspicious activity effectively and minimize potential financial losses.

## 📌 Project Overview

- **Goal:** Accurately classify credit card transactions as fraudulent or legitimate.
- **Dataset:** European credit card transactions (from Kaggle).
- **Tech Stack:** Python, Pandas, Scikit-Learn, Matplotlib, Seaborn

---

## 🚀 Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA) and visualization
- Feature selection and scaling
- Model training using multiple ML algorithms
- Evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
- Handling imbalanced data using SMOTE and undersampling techniques
- Confusion matrix and ROC curve visualization

---

## 📂 Folder Structure

```
Credit_Card_Fraud_Detection_System/
│
├── data/                        # Dataset files (CSV)
├── models/                      # Saved model files
├── notebooks/                   # Jupyter notebooks
├── visuals/                     # Plots and graphs
├── fraud_detection.py          # Main Python script
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

---

## 📈 Evaluation Metrics

- **Confusion Matrix**
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC-AUC Curve**

---

## 📊 Algorithms Used

- Logistic Regression
- Random Forest Classifier
- Decision Trees
- XGBoost
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

---

## 📚 Future Improvements

- Integrate deep learning (LSTM or autoencoders)
- Real-time fraud detection with stream data
- Web interface or dashboard using Flask/Streamlit


## 🤝 Acknowledgements

- [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Scikit-learn, Matplotlib, Seaborn, Imbalanced-learn
