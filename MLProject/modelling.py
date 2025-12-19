import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

os.environ['MLFLOW_TRACKING_USERNAME'] = 'hqee' 
os.environ['MLFLOW_TRACKING_PASSWORD'] = '46bca6ad5b7704ceda1a802e3009993a9bbaab0e'

dagshub_url = "https://dagshub.com/hqee/eksperimen_sml_haqi-dhiya.mlflow"
mlflow.set_tracking_uri(dagshub_url)

mlflow.sklearn.autolog()

df = pd.read_csv('diabetes_preprocessed.csv')
X = df.drop(columns=['Outcome'])
y = df['Outcome']

mlflow.set_experiment("diabetes_Prediction")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Evaluasi
predictions = model.predict(X)
acc = accuracy_score(y, predictions)

print(f"Model (Autolog via MLflow Project) berhasil dilatih dengan akurasi: {acc}")