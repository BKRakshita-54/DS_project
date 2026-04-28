import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv("student_subject_dataset.csv")

X = df[["credits", "studytime", "absences", "failures", "internal1", "internal2"]]
y = df["final_score"]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained and saved!")