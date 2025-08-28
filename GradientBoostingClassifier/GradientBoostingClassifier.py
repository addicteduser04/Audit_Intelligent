import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
# Load data
book1 = pd.read_csv('./../data/Book1.csv', low_memory=False)
features = pd.read_csv('./../data/features.csv', low_memory=False)
new_features = features[['CodeClient','CompteProduit','CentreAnalyse']].copy()


# --- Gradient Boosting ---
X2 = new_features.copy()
y2 = features['in_book2']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
gb_model2 = GradientBoostingClassifier(random_state=42)
gb_model2.fit(X2_train, y2_train)
pred2_gb = gb_model2.predict(X2_test)
print('Book2 GradientBoosting Accuracy:', accuracy_score(y2_test, pred2_gb))
print(classification_report(y2_test, pred2_gb))


# Repeat for Book3
X3 = new_features.copy()
y3 = features['in_book3']
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)

# --- Gradient Boosting ---
gb_model3 = GradientBoostingClassifier(random_state=42)
gb_model3.fit(X3_train, y3_train)
pred3_gb = gb_model3.predict(X3_test)
print('Book3 GradientBoosting Accuracy:', accuracy_score(y3_test, pred3_gb))
print(classification_report(y3_test, pred3_gb))

joblib.dump(gb_model2, "gb_model2.pkl")
joblib.dump(gb_model3, "gb_model3.pkl")


# Save predictions for Book2 
results_book2 = pd.DataFrame({
    'NumFacture': book1.loc[X2_test.index, 'NumFacture'],
    'Book2_Predicted_GB': pred2_gb,
    'Book2_Actual': y2_test.values
})
results_book2.to_csv('transaction_match_predictions_book2_GBC.csv', index=False)

# Save predictions for Book3 
results_book3 = pd.DataFrame({
    'NumFacture': book1.loc[X3_test.index, 'NumFacture'],
    'Book3_Predicted_GB': pred3_gb,
    'Book3_Actual': y3_test.values
})
results_book3.to_csv('transaction_match_predictions_book3_GBC.csv', index=False)
