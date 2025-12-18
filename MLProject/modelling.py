import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow
import mlflow.sklearn
import dagshub

dagshub.init(repo_owner='hqee', repo_name='eksperimen_sml_haqi-dhiya', mlflow=True)

# Load Data Preprocessed
df = pd.read_csv('diabetes_preprocessed.csv')
X = df.drop(columns=['Outcome'])
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Setup MLflow Experiment
mlflow.set_experiment("Diabetes_Classification_Advanced")

with mlflow.start_run(run_name="RandomForest_Tuning"):
    # Hyperparameter Tuning
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Log Best Params
    mlflow.log_params(grid_search.best_params_)
    
    # Log Metrics
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    
    # Artefak 1: Confusion Matrix Image
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    # Artefak 2: Classification Report Text
    report = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")
    
    # Log Model
    mlflow.sklearn.log_model(best_model, "diabetes_model_final")
    
    print(f"Eksperimen selesai. Akurasi: {acc}")
    print("Metadata telah dikirim ke DagsHub!")