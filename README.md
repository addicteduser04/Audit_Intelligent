# Audit Intelligent

Audit Intelligent is a Python-based project designed to detect anomalies in accounting journals using machine learning.  
It applies several classification algorithms to identify suspicious or unusual transactions based on the Moroccan accounting plan.

---

## Project Overview

The main goal of Audit Intelligent is to assist auditors by automatically identifying potential irregularities in accounting journals.  
It combines data preprocessing, statistical analysis, and machine learning models to classify accounting entries as normal or anomalous.

---

## Repository Structure

├── data/
│ ├── raw/ ← Contains the raw accounting data (Book1 original files)
│ ├── Book1_filtered.csv ← Cleaned and preprocessed version of the data
│ ├── Book1_scaled.csv ← Scaled or encoded dataset used for model training
│ └── ... ← Other derived data files
│
├── DecisionTreeClassifier/
│ ├── decision_tree.py ← Code used to train and test the Decision Tree model
│ ├── decision_tree_model.pkl ← Saved trained model
│ ├── decision_tree_model2.pkl ← Alternative trained model
│ ├── Y_predicted.csv ← Predicted Y values for test data
│ └── Y_predicted2.csv ← Additional predicted results
│
├── RandomForestClassifier/
│ ├── random_forest.py
│ ├── rf_model.pkl
│ ├── rf_model2.pkl
│ ├── Y_predicted.csv
│ └── Y_predicted2.csv
│
├── GradientBoostingClassifier/
│ ├── gradient_boosting.py
│ ├── gb_model.pkl
│ ├── gb_model2.pkl
│ ├── Y_predicted.csv
│ └── Y_predicted2.csv
│
├── LogisticRegression/
│ ├── logistic_regression.py
│ ├── lr_model.pkl
│ ├── lr_model2.pkl
│ ├── Y_predicted.csv
│ └── Y_predicted2.csv
│
├── KNeighborsClassifier/
│ ├── knn.py
│ ├── knn_model.pkl
│ ├── knn_model2.pkl
│ ├── Y_predicted.csv
│ └── Y_predicted2.csv
│
├── data_preprocessing.py ← Script to clean and prepare data before training
├── data_understanding.py ← Script for data exploration and visualization
├── dockerfile ← Docker container configuration
├── requirements.txt ← Python dependencies
└── README.md ← Project documentation



---

## How It Works

1. **Data Understanding**  
   - Raw accounting journals are stored in the `data/` folder.  
   - The `data_understanding.py` script explores and analyzes the data (missing values, correlations, etc.).

2. **Data Preprocessing**  
   - Run `data_preprocessing.py` to clean, encode, and scale the data.  
   - The processed datasets (such as `Book1_filtered.csv` and `Book1_scaled.csv`) are saved inside `data/`.

3. **Model Training**  
   - Each machine learning folder contains:
     - A Python script for model training and testing.
     - Two `.pkl` files storing trained models.
     - Two `.csv` files storing the predicted Y values.

---

## Machine Learning Models

| Algorithm | Folder | Description |
|------------|---------|-------------|
| Decision Tree | `DecisionTreeClassifier/` | Interpretable tree-based model |
| Random Forest | `RandomForestClassifier/` | Ensemble of decision trees for better generalization |
| Gradient Boosting | `GradientBoostingClassifier/` | High-performance boosting algorithm |
| Logistic Regression | `LogisticRegression/` | Linear baseline classifier |
| K-Nearest Neighbours | `KNeighborsClassifier/` | Distance-based approach for anomaly detection |

---

## Installation and Usage

### 1. Clone the Repository
git clone https://github.com/addicteduser04/Audit_Intelligent.git
cd Audit_Intelligent
2. Install Dependencies
pip install -r requirements.txt

3. Run Data Preprocessing
python data_preprocessing.py

4. Train and Test a Model
Example with Decision Tree:
python DecisionTreeClassifier/decision_tree.py

5. Evaluate Models
python testmodel.py

6. Optional: Run via Docker
docker build -t audit_intelligent .
docker run -it audit_intelligent

Future Improvements

Development of a web interface (Streamlit or Flask) for audit visualization.

Integration with MySQL for persistent data storage.

Implementation of explainability tools (e.g., SHAP, LIME).

Real-time anomaly detection and continuous auditing.

Improved ensemble and hybrid detection techniques.

Author

Sifeddine Elkadiri
Email: elkadirisifeddine@gmail.com


License

This project is open-source and available for educational and research purposes.
