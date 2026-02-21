import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

os.makedirs("model", exist_ok=True)

data = pd.read_csv("heart.csv")

X = data.drop("target",axis=1)
y = data["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_scaled,y)

joblib.dump(model,"model/ml_model.pkl")
joblib.dump(scaler,"model/scaler.pkl")

print("✅ ML Model Saved")
