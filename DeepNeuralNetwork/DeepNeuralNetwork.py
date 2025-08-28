import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from keras.models import Sequential
from keras.layers import Dense

# Load data
book1 = pd.read_csv('./data/Book1.csv', low_memory=False)
features = pd.read_csv('./data/features.csv', low_memory=False)
new_features = features[['CodeClient','CompteProduit','CentreAnalyse']].copy()

# --- Deep Neural Network ---
X2 = new_features.copy()
y2 = features['in_book2']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

model2 = Sequential([
    Dense(32, activation='relu', input_shape=(X2_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model2.fit(X2_train, y2_train, epochs=30, batch_size=32, verbose=0)
pred2_dnn = (model2.predict(X2_test) > 0.5).astype(int).flatten()
print('Book2 DNN Accuracy:', accuracy_score(y2_test, pred2_dnn))
print(classification_report(y2_test, pred2_dnn))

# Repeat for Book3
X3 = new_features.copy()
y3 = features['in_book3']
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)

model3 = Sequential([
    Dense(32, activation='relu', input_shape=(X3_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model3.fit(X3_train, y3_train, epochs=30, batch_size=32, verbose=0)
pred3_dnn = (model3.predict(X3_test) > 0.5).astype(int).flatten()
print('Book3 DNN Accuracy:', accuracy_score(y3_test, pred3_dnn))
print(classification_report(y3_test, pred3_dnn))

model2.save("dnn_model2.h5")
model3.save("dnn_model3.h5")

# Save predictions for Book2 
results_book2 = pd.DataFrame({
    'NumFacture': book1.loc[X2_test.index, 'NumFacture'],
    'Book2_Predicted_DNN': pred2_dnn,
    'Book2_Actual': y2_test.values
})
results_book2.to_csv('transaction_match_predictions_book2_DNN.csv', index=False)

# Save predictions for Book3 
results_book3 = pd.DataFrame({
    'NumFacture': book1.loc[X3_test.index, 'NumFacture'],
    'Book3_Predicted_DNN': pred3_dnn,
    'Book3_Actual': y3_test.values
})
results_book3.to_csv('transaction_match_predictions_book3_DNN.csv', index=False)
