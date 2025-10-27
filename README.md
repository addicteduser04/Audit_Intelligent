# Audit Intelligent

Audit Intelligent is a Python-based project designed to detect anomalies in accounting journals using machine learning.  
It applies several classification algorithms to identify suspicious or unusual transactions based on the Moroccan accounting plan.

---

## Project Overview

The main goal of Audit Intelligent is to assist auditors by automatically identifying potential irregularities in accounting journals.  
It combines data preprocessing, statistical analysis, and machine learning models to classify accounting entries as normal or anomalous.

---
```

## Repository Structure

Audit_Intelligent/
│
├── data/
│ ├── Book1.csv #original file AFRIWARE
│ ├── Book2.csv #original file JDE_911
│ ├── Book3.csv #original file JDE_3b11
│ ├── filtered_book5.csv # Cleaned & preprocessed dataset
│ ├── numfacture_counts.csv # Derived data used for analysis
│ └── ... # Other processed files
│
├── DecisionTreeClassifier/
│ ├── decision_tree.py # Decision Tree model code
│ ├── dt_model2.pkl # Trained model version 2
│ ├── dt_model3.pkl # Trained model version 3
│ ├── transaction_match_predictions_book2_DT.csv # Predictions for Book2
│ └── ... # Other model results
│
├── LogisticRegression/
│ ├── logistic_regression.py # Logistic Regression model code
│ ├── lr_model2.pkl # Trained model version 2
│ ├── lr_model3.pkl # Trained model version 3
│ ├── transaction_match_predictions_book2_LR.csv # Predictions for Book2
│ └── transaction_match_predictions_book3_LR.csv # Predictions for Book3
│
├── GradientBoostingClassifier/
│ ├── gradient_boosting.py # Gradient Boosting model code
│ └── ... # Model files & predictions
│
├── KNeighborsClassifier/
│ ├── knn.py # KNN model code
│ └── ... # Model files & predictions
│
├── RandomForestClassifier/
│ ├── random_forest.py # Random Forest model code
│ └── ... # Model files & predictions
├── Report/
│ └── EL KADIRI SIFEDDINE.pdf # Report in french 
|
├── Figure_1.png # Correlation matrix for feature selection
│
├── data_preprocessing.py # Script for data cleaning & preparation
├── data_understanding.py # Script for exploratory data analysis
├── dockerfile # Docker configuration file
└── README.md # Project documentation

```


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
