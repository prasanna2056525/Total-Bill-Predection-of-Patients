
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def init():
    global model, encoders
    model = joblib.load('model.pkl')

    # Rebuild encoders exactly as in training
    encoders = {}
    categorical_columns = {
        "Gender": ["Female", "Male"],
        "Medical_Condition": ["Arthritis", "Asthma", "Cancer", "Diabetes", "Hypertension", "Obesity"],
        "Admission_Type": ["Elective", "Emergency", "Urgent"],
        "Insurance_Provider": ["Aetna", "Blue Cross", "Cigna", "Medicaid", "Medicare", "UnitedHealthcare"]
    }
    for col, classes in categorical_columns.items():
        le = LabelEncoder()
        le.fit(classes)
        encoders[col] = le

def run(raw_data):
    try:
        data = json.loads(raw_data)
        age = data['Age']
        gender = encoders['Gender'].transform([data['Gender']])[0]
        condition = encoders['Medical_Condition'].transform([data['Medical_Condition']])[0]
        admission = encoders['Admission_Type'].transform([data['Admission_Type']])[0]
        insurance = encoders['Insurance_Provider'].transform([data['Insurance_Provider']])[0]
        los = data['Length_of_Stay']

        input_df = pd.DataFrame([{
            "Age": age,
            "Gender_enc": gender,
            "Medical_Condition_enc": condition,
            "Admission_Type_enc": admission,
            "Insurance_Provider_enc": insurance,
            "Length_of_Stay": los
        }])

        prediction = model.predict(input_df)[0]
        return json.dumps({'prediction': round(prediction, 2)})
    except Exception as e:
        return json.dumps({'error': str(e)})
