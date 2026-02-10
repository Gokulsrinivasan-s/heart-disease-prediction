from flask import Flask, request, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model directly (because you saved only model)
model = joblib.load("heart_model.pkl")

# IMPORTANT: Must match training columns exactly
columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalch", "exang",
    "oldpeak", "slope", "ca", "thal"
]

numeric_fields = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {}

        for col in columns:
            value = request.form[col]

            if col in numeric_fields:
                input_data[col] = float(value)
            else:
                input_data[col] = value

        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]

        if prediction == 1:
            result = "Heart Disease Detected"
        else:
            result = "No Heart Disease"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
