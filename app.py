from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and dataset
model = pickle.load(open("model.pkl", "rb"))
df = pd.read_csv("student_subject_dataset.csv")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analysis")
def analysis():
    table = df.head(10).to_html(index=False)
    return render_template("analysis.html", table=table)

@app.route("/explain")
def explain():
    return render_template("explain.html")

@app.route("/predict", methods=["POST"])
def predict():
    subject = request.form["subject"]
    credits = int(request.form["credits"])
    studytime = float(request.form["studytime"])
    absences = float(request.form["absences"])
    failures = float(request.form["failures"])
    internal1 = float(request.form["internal1"])
    internal2 = float(request.form["internal2"])
    interest = int(request.form["interest"])  # NEW

    # Prediction
    features = np.array([[credits, studytime, absences, failures, internal1, internal2]])
    prediction = model.predict(features)[0]

    # Explanation
    reasons = []
    if studytime < 2:
        reasons.append("low study time")
    if absences > 5:
        reasons.append("high absences")
    if failures > 0:
        reasons.append("weak fundamentals")
    if internal1 < 50 or internal2 < 50:
        reasons.append("low internal scores")
    if credits >= 4:
        reasons.append("high subject difficulty")
    if interest == 1:
        reasons.append("low interest in subject")
    elif interest == 3:
        reasons.append("high interest improves performance")

    explanation = f"For {subject}, performance is affected due to {', '.join(reasons)}." if reasons else "Performance is stable."

    # Suggestions
    suggestions = []
    if credits >= 4:
        suggestions.append("Allocate more time for this subject")
    if studytime < 2:
        suggestions.append("Increase study time")
    if absences > 5:
        suggestions.append("Reduce absences")
    if failures > 0:
        suggestions.append("Revise fundamentals")
    if internal1 < 50 or internal2 < 50:
        suggestions.append("Improve internal test preparation")
    if interest == 1:
        suggestions.append("Try to build interest using videos or practical learning")

    # Contributions
    internal_avg = (internal1 + internal2) / 2

    contributions = {
        "Internal Performance": round(internal_avg, 2),
        "Study Time": round(studytime * 5, 2),
        "Interest Level": round(interest * 4, 2),
        "Absences": round(-absences * 1.5, 2),
        "Failures": round(-failures * 5, 2),
        "Subject Difficulty": round(-(credits - 2) * 3, 2)
    }

    return render_template("result.html",
                           subject=subject,
                           prediction=round(prediction, 2),
                           explanation=explanation,
                           suggestions=suggestions,
                           contributions=contributions)

if __name__ == "__main__":
    app.run(debug=True)