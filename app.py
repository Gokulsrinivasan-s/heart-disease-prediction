from flask import Flask, request, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ---- Load model file ----
model_path = "heart_model.pkl"
model_dict = joblib.load(model_path)

# Extract objects
model = model_dict['model']
imputer = model_dict.get('imputer')
encoders = model_dict.get('encoders', {})
columns = list(model_dict.get('columns', []))

# ---- Categorical dropdowns ----
dropdowns = {
    "sex": ["Male", "Female"],
    "dataset": ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"],
    "cp": ["typical angina", "asymptomatic", "non-anginal", "atypical angina"],
    "fbs": ["No", "Yes"],
    "restecg": ["Normal", "ST-T Abnormality", "LV Hypertrophy"],
    "exang": ["No", "Yes"],
    "slope": ["Upsloping", "Flat", "Downsloping"],
    "thal": ["Normal", "Fixed Defect", "Reversible Defect"]
}

numeric_fields = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]

# ---- Mapping ----
mapping = {
    "fbs": {"Yes": True, "No": False},
    "exang": {"Yes": True, "No": False},
    "restecg": {
        "Normal": "normal",
        "ST-T Abnormality": "st-t abnormality",
        "LV Hypertrophy": "lv hypertrophy"
    },
    "slope": {
        "Upsloping": "upsloping",
        "Flat": "flat",
        "Downsloping": "downsloping"
    },
    "thal": {
        "Normal": "normal",
        "Fixed Defect": "fixed defect",
        "Reversible Defect": "reversable defect"
    },
    "cp": {
        "typical angina": "typical angina",
        "asymptomatic": "asymptomatic",
        "non-anginal": "non-anginal",
        "atypical angina": "atypical angina"
    },
    "sex": {"Male": "Male", "Female": "Female"},
    "dataset": {
        "Cleveland": "Cleveland",
        "Hungary": "Hungary",
        "Switzerland": "Switzerland",
        "VA Long Beach": "VA Long Beach"
    }
}

# ---- Home Route ----
@app.route('/')
def home():
    return render_template(
        'index.html',
        columns=columns,
        dropdowns=dropdowns,
        numeric_fields=numeric_fields
    )

# ---- Prediction Route ----
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {}

        # Read form data
        for col in columns:
            val = request.form[col]
            if col in numeric_fields:
                input_data[col] = float(val)
            else:
                input_data[col] = val

        input_df = pd.DataFrame([input_data])

        # Apply mapping
        for col, map_dict in mapping.items():
            if col in input_df.columns:
                input_df[col] = input_df[col].map(map_dict)

        # Apply encoders
        for col, le in encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))

        # Apply imputer
        if imputer:
            input_df_processed = imputer.transform(input_df)
        else:
            input_df_processed = input_df.values

        # Predict
        prediction = model.predict(input_df_processed)[0]
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        return render_template(
            'index.html',
            columns=columns,
            dropdowns=dropdowns,
            numeric_fields=numeric_fields,
            prediction_text=result
        )

    except Exception as e:
        return f"Error: {str(e)}"

# ---- For Render Deployment ----
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
