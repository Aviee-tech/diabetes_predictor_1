from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__, static_folder='static')

bundle = joblib.load('best_diabetes_model.joblib')
model = bundle['model']
scaler = bundle['scaler']
feature_names = bundle['feature_names']
metrics = bundle['test_metrics']

def engineer_features(df):
    d = df.copy()
    d['Glucose_BMI'] = d['Glucose'] * d['BMI']
    d['Glucose_Age'] = d['Glucose'] * d['Age']
    d['Insulin_Glucose_Ratio'] = d['Insulin'] / (d['Glucose'] + 1)
    d['BMI_Age'] = d['BMI'] * d['Age']
    d['Glucose_Category'] = pd.cut(d['Glucose'], bins=[0,99,125,200,300], labels=[0,1,2,3]).astype(int)
    d['BMI_Category'] = pd.cut(d['BMI'], bins=[0,18.5,24.9,29.9,34.9,100], labels=[0,1,2,3,4]).astype(int)
    d['Age_Group'] = pd.cut(d['Age'], bins=[0,30,45,60,100], labels=[0,1,2,3]).astype(int)
    d['BP_Category'] = pd.cut(d['BloodPressure'], bins=[0,80,89,99,200], labels=[0,1,2,3]).astype(int)
    d['Glucose_Squared'] = d['Glucose'] ** 2
    d['BMI_Squared'] = d['BMI'] ** 2
    d['Age_Squared'] = d['Age'] ** 2
    d['DiabetesRiskScore'] = (
        (d['Glucose'] > 125).astype(int) * 3 +
        (d['BMI'] > 30).astype(int) * 2 +
        (d['Age'] > 45).astype(int) * 2 +
        (d['Pregnancies'] > 4).astype(int) * 1 +
        (d['DiabetesPedigreeFunction'] > 0.5).astype(int) * 2 +
        (d['Insulin'] > 140).astype(int) * 1
    )
    d['SkinThickness_BMI'] = d['SkinThickness'] / (d['BMI'] + 0.1)
    d['Pedigree_Age'] = d['DiabetesPedigreeFunction'] * d['Age']
    return d

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        raw = pd.DataFrame([{
            'Pregnancies': float(data['pregnancies']),
            'Glucose': float(data['glucose']),
            'BloodPressure': float(data['blood_pressure']),
            'SkinThickness': float(data['skin_thickness']),
            'Insulin': float(data['insulin']),
            'BMI': float(data['bmi']),
            'DiabetesPedigreeFunction': float(data['diabetes_pedigree']),
            'Age': float(data['age']),
            'Outcome': 0
        }])
        feat = engineer_features(raw).drop('Outcome', axis=1)
        feat = feat[feature_names]
        feat_scaled = scaler.transform(feat)
        prob = float(model.predict_proba(feat_scaled)[0][1])
        pred = 1 if prob >= 0.5 else 0
        return jsonify({
            'prediction': pred,
            'probability': round(prob * 100, 1),
            'label': 'HIGH RISK' if pred == 1 else 'LOW RISK',
            'model_accuracy': round(metrics['accuracy'] * 100, 1),
            'model_auc': round(metrics['roc_auc'], 3)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model_info')
def model_info():
    return jsonify({
        'model_name': bundle['model_name'],
        'accuracy': round(metrics['accuracy'] * 100, 1),
        'roc_auc': round(metrics['roc_auc'], 3),
        'f1': round(metrics['f1'], 3)
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
