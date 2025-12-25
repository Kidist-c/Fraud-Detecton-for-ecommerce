🛡️ Fraud Detection System
📌 Project Overview

This project focuses on building a machine learning–based fraud detection system using transactional data. The goal is to accurately identify fraudulent transactions while addressing key challenges such as extreme class imbalance, high false-negative cost, and model interpretability.

The project is implemented using modular, reusable, and scalable components suitable for both credit card and e-commerce fraud datasets.
#----------------------------------------------------------------------

🛡️ Fraud Detection System
📌 Project Overview

This project focuses on building a machine learning–based fraud detection system using transactional data. The goal is to accurately identify fraudulent transactions while addressing key challenges such as extreme class imbalance, high false-negative cost, and model interpretability.

The project is implemented using modular, reusable, and scalable components suitable for both credit card and e-commerce fraud datasets.

#-------------------------------------------------------------------------------------------------------

📂 Project Structure
Fraud-Detecton-for-ecommerce/
│
├── data/
│ ├── raw/ # Original datasets
│ ├── processed/ # Train-test splits and cleaned data
│
├── notebooks/
│ ├── EDA.ipynb # Exploratory Data Analysis
│ ├── FeatureEngineering.ipynb
│ ├── Modeling.ipynb # Model training & evaluation
│
├── src/
│ ├── creditcard_transformer.py # Data preprocessing & SMOTE
│ ├── model_trainer.py # Model training & evaluation
│
├──
│ # EDA plots & visualizations
│
├── README.md
├── requirements.txt
└── .gitignore

#-----------------------------------------------------------------------------------------------------------------
📊 Datasets Used

Credit Card Dataset

Target column: Class

E-commerce Fraud Dataset

Target column: class

Both datasets exhibit severe class imbalance, which is a central challenge addressed in this project.

🔍 Exploratory Data Analysis (EDA)

Key insights from EDA include:

The vast majority of transactions are low-value

Fraudulent transactions tend to cluster in lower transaction amounts

Transactions are concentrated within a short time window

Fraud patterns are highly skewed, justifying imbalance-aware modeling

📌 Business implication:
Missing fraudulent transactions (false negatives) has a much higher cost than false positives, motivating the use of recall-sensitive metrics.

#---------------------------------------------------------------------------------------------------------------------------------------

⚙️ Data Preprocessing & Feature Engineering

Checked for missing values and data consistency

Standardized numerical features where required

Created time-based and behavior-related features

Applied SMOTE only on the training set to avoid data leakage

Used stratified splitting to preserve class distribution

⚖️ Handling Class Imbalance

Class imbalance was addressed using:

SMOTE (Synthetic Minority Over-sampling Technique) for training data

class_weight="balanced" for Logistic Regression

Evaluation metrics robust to imbalance (PR-AUC, F1)

📌 Why this matters:
Imbalanced data can lead to models that appear accurate but completely fail to detect fraud

Imbalanced data can lead to models that appear accurate but completely fail to detect fraud.

🤖 Model Building
1️⃣ Baseline Model — Logistic Regression

Interpretable and fast

Serves as a performance benchmark

Uses class-weight balancing

2️⃣ Ensemble Model — Random Forest

Captures non-linear relationships

More robust to noise and feature interactions

Hyperparameter tuning applied using RandomizedSearchCV

📈 Model Evaluation Metrics

The following metrics are used due to the imbalanced nature of fraud data:

PR-AUC (Precision-Recall AUC) – primary metric

F1-Score – balance between precision and recall

Confusion Matrix – error analysis

🔁 Cross-Validation

Stratified K-Fold (k=5) used

Reports mean and standard deviation for:

PR-AUC

F1-Score

📌 This ensures performance stability and reduces overfitting risk.

🧪 Model Comparison & Selection

Models are compared side-by-side based on:

PR-AUC performance

F1-Score

Interpretability vs complexity trade-off

The selected model balances fraud detection performance and operational interpretability.

🧠 Key Learnings

Accuracy is misleading for fraud detection

PR-AUC is more informative than ROC-AUC

Proper handling of class imbalance is critical

Modular code greatly improves reusability and debugging

Cross-validation is essential for reliable evaluation

🚧 Current Limitations

Feature importance analysis can be expanded

Explainability tools (e.g., SHAP) not yet integrated

Pipeline-based deployment not yet implemented

🚀 Next Steps

Integrate preprocessing and modeling into a unified pipeline

Add model explainability (SHAP)

Improve error handling and logging

Enhance documentation and unit testing

Explore gradient boosting models (XGBoost / LightGBM)

🛠️ Technologies Used

Python

NumPy, Pandas

Scikit-learn

Imbalanced-learn

Matplotlib / Seaborn
