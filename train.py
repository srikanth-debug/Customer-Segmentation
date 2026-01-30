import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
mlflow.set_tracking_uri("sqlite:///mlflow.db")


df = pd.read_csv("rfm_final.csv")

X = df[['Frequency','Monetary','Tenure','AvgOrderValue']]
y = df['Churn']

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier(
    n_estimators = 200,
    max_depth = 6,
    random_state = 42
)

mlflow.set_experiment("Churn_Prediction")

with mlflow.start_run():
    model.fit(X_train,y_train)

    preds = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test,preds)

    mlflow.log_metric("roc_auc",auc)
    mlflow.log_params({
        "n_estimators": 200,
        "max_depth" : 6
    })

    mlflow.sklearn.log_model(model, name = "churn_model")

    print("ROC AUC :",auc)