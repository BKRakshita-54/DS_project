import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("student_subject_dataset.csv")

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

# 🔹 Correlation Heatmap
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.savefig("static/corr.png", bbox_inches='tight')
plt.close()

# 🔹 Feature Importance
from sklearn.ensemble import RandomForestRegressor

X = df[["credits", "studytime", "absences", "failures", "internal1", "internal2"]]
y = df["final_score"]

model = RandomForestRegressor()
model.fit(X, y)

importances = model.feature_importances_

plt.figure(figsize=(6,4))
plt.barh(X.columns, importances)
plt.title("Feature Importance")
plt.savefig("static/importance.png", bbox_inches='tight')
plt.close()

print("Visuals generated successfully!")