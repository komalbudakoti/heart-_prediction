from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Ensure the model filename matches your saved model
model = pickle.load(open("heart_pred_Model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = float(request.form["age"])
    sex = float(request.form["sex"])
    cp = float(request.form["chest_pain_type"])
    trestbps = float(request.form["resting_bp_s"])
    chol = float(request.form["cholesterol"])
    fbs = float(request.form["fasting_blood_sugar"])
    restecg = float(request.form["resting_ecg"])
    thalach = float(request.form["max_heart_rate"])
    exang = float(request.form["exercise_angina"])
    oldpeak = float(request.form["oldpeak"])
    slope = float(request.form["ST_slope"])

    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "The patient is likely to have heart disease."
    else:
        result = "The patient is unlikely to have heart disease."
    
    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)