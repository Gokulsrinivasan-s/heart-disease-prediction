from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# ---- Load model dictionary ---
model_path = "heart_model.pkl"
model_dict = joblib.load(model_path)



model_dict = joblib.load(model_path)

# Extract objects
model = model_dict['model']
imputer = model_dict.get('imputer')
encoders = model_dict.get('encoders', {})
columns = list(model_dict.get('columns', []))

# ---- Categorical dropdowns (user-friendly) ----
dropdowns = {
    "sex": ["Male", "Female"],
    "dataset": ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"],
    "cp": ["typical angina", "asymptomatic", "non-anginal", "atypical angina"],
    "fbs": ["No", "Yes"],         # Boolean
    "restecg": ["Normal", "ST-T Abnormality", "LV Hypertrophy"],
    "exang": ["No", "Yes"],       # Boolean
    "slope": ["Upsloping", "Flat", "Downsloping"],
    "thal": ["Normal", "Fixed Defect", "Reversible Defect"]
}

# Numeric fields
numeric_fields = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]

# ---- Mapping to encoder expected values ----
mapping = {
    # Booleans
    "fbs": {"Yes": True, "No": False},
    "exang": {"Yes": True, "No": False},
    # Other categorical
    "restecg": {"Normal": "normal", "ST-T Abnormality": "st-t abnormality", "LV Hypertrophy": "lv hypertrophy"},
    "slope": {"Upsloping": "upsloping", "Flat": "flat", "Downsloping": "downsloping"},
    "thal": {"Normal": "normal", "Fixed Defect": "fixed defect", "Reversible Defect": "reversable defect"},
    "cp": {"typical angina": "typical angina", "asymptomatic": "asymptomatic", 
           "non-anginal": "non-anginal", "atypical angina": "atypical angina"},
    "sex": {"Male": "Male", "Female": "Female"},
    "dataset": {"Cleveland":"Cleveland", "Hungary":"Hungary", "Switzerland":"Switzerland", "VA Long Beach":"VA Long Beach"}
}

# ---- Home route ----
@app.route('/')
def home():
    return render_template('index.html', columns=columns, dropdowns=dropdowns, numeric_fields=numeric_fields)

# ---- Predict route ----
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1️⃣ Read form inputs
        input_data = {}
        for col in columns:
            val = request.form[col]
            if col in numeric_fields:
                input_data[col] = float(val)
            else:
                input_data[col] = val

        input_df = pd.DataFrame([input_data])

        # 2️⃣ Map form input to what encoders expect
        for col, map_dict in mapping.items():
            if col in input_df.columns:
                input_df[col] = input_df[col].map(map_dict)

        # 3️⃣ Apply LabelEncoders
        for col, le in encoders.items():
            if col in input_df.columns:
                # Only convert non-boolean columns to str
                if input_df[col].dtype == 'bool':
                    input_df[col] = le.transform(input_df[col])
                else:
                    input_df[col] = le.transform(input_df[col].astype(str))

        # 4️⃣ Apply imputer if exists
        if imputer:
            input_df_processed = imputer.transform(input_df)
        else:
            input_df_processed = input_df.values

        # 5️⃣ Predict
        prediction = model.predict(input_df_processed)[0]
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        return render_template('index.html', columns=columns, dropdowns=dropdowns, numeric_fields=numeric_fields, prediction_text=result)

    except Exception as e:
        return f"Error: {str(e)}"

# ---- Run app ----
if __name__ == "__main__":
    app.run(debug=True)
