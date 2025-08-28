import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Load data
book1 = pd.read_csv('./../data/Book1.csv', low_memory=False)
features = pd.read_csv('./../data/features.csv', low_memory=False)
new_features = features[['CodeClient','CompteProduit','CentreAnalyse']].copy()

# Model for Book2
X2 = new_features.copy()
y2 = features['in_book2']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
model2 = RandomForestClassifier(random_state=42)
model2.fit(X2_train, y2_train)
pred2 = model2.predict(X2_test)
print('Book2 RandomForestClassifier Transaction Match Accuracy:', accuracy_score(y2_test, pred2))
print(classification_report(y2_test, pred2))

# Model for Book3
X3 = new_features.copy()
y3 = features['in_book3']
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)
model3 = RandomForestClassifier(random_state=42)
model3.fit(X3_train, y3_train)
pred3 = model3.predict(X3_test)
print('Book3 RandomForestClassifier Transaction Match Accuracy:', accuracy_score(y3_test, pred3))
print(classification_report(y3_test, pred3))

joblib.dump(model2, "rf_model2.pkl")
joblib.dump(model3, "rf_model3.pkl")


# Save predictions for Book2
results_book2 = pd.DataFrame({
    'NumFacture': book1.loc[X2_test.index, 'NumFacture'],
    'Book2_Predicted': pred2,
    'Book2_Actual': y2_test.values
})
results_book2.to_csv('transaction_match_predictions_book2_RFC.csv', index=False)

# Save predictions for Book3
results_book3 = pd.DataFrame({
    'NumFacture': book1.loc[X3_test.index, 'NumFacture'],
    'Book3_Predicted': pred3,
    'Book3_Actual': y3_test.values
})
results_book3.to_csv('transaction_match_predictions_book3_RFC.csv', index=False)
