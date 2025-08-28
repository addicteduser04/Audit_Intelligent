import pandas as pd 
import joblib 

book1=pd.read_csv('./data/Book1.csv')
book2=pd.read_csv('./data/Book2.csv')
book3=pd.read_csv('./data/Book3.csv')

gb_model2=joblib.load("./GradientBoostingClassifier/gb_model2.pkl")
gb_model3=joblib.load("./GradientBoostingClassifier/gb_model3.pkl")


y_pred2=gb_model2.predict(book1.drop(columns=['TypeFacture','NumLigne', 'DateFacture', 'DateCreation', 'DateModification', 'DateEDI', 'ReferenceEDI', 'Taxes'], errors='ignore').apply(pd.to_numeric, errors='coerce').fillna(0))
y_pred3=gb_model3.predict(book1.drop(columns=['TypeFacture','NumLigne', 'DateFacture', 'DateCreation', 'DateModification', 'DateEDI', 'ReferenceEDI', 'Taxes'], errors='ignore').apply(pd.to_numeric, errors='coerce').fillna(0))

# Save y_pred2 as CSV
results2 = pd.DataFrame({
	'NumFacture': book1['NumFacture'],
	'y_pred': y_pred2
})

nbre=results2['y_pred'].sum()
dim=results2.shape[0]
print(f"Total predicted transactions in Book2: {nbre}")
print(f"Total transactions in Book1: {dim}")
diff=dim - nbre
print(f"le nombre de transactions dans book1 exclu book2 {diff}")
print((diff/dim)*100)
results2.to_csv('y_pred2_results.csv', index=False)

# Save y_pred3 as CSV
results3 = pd.DataFrame({
	'NumFacture': book1['NumFacture'],
	'y_pred': y_pred3
})
results3.to_csv('y_pred3_results.csv', index=False)

