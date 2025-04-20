from flask import Flask, render_template, request, jsonify
import os
import joblib
import numpy as np
from model.heart_model import HeartDiseaseModel, preprocess_input

app = Flask(__name__)

# Initialize model during startup instead of using deprecated decorator
@app.before_request
def check_model():
    """Check if model exists before processing any request"""
    model_path = os.path.join('model', 'model.pkl')
    if not os.path.exists(model_path):
        print("Training and saving model...")
        os.makedirs('model', exist_ok=True)
        heart_model = HeartDiseaseModel()
        heart_model.train()
        heart_model.save_model(model_path)
        print("Model saved successfully!")

# Load the trained model
def get_model():
    model_path = os.path.join('model', 'model.pkl')
    return joblib.load(model_path)

@app.route('/')
def index():
    """Render the main page with the input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the form submission and return prediction"""
    try:
        # Get data from form
        form_data = request.form.to_dict()
        
        # Process the input data
        input_features = {
            'age': int(form_data.get('age')),
            'sex': int(form_data.get('sex')),
            'cp': int(form_data.get('cp')),
            'trestbps': int(form_data.get('trestbps')),
            'chol': int(form_data.get('chol')),
            'fbs': int(form_data.get('fbs')),
            'restecg': int(form_data.get('restecg')),
            'thalach': int(form_data.get('thalach')),
            'exang': int(form_data.get('exang')),
            'oldpeak': float(form_data.get('oldpeak')),
            'slope': int(form_data.get('slope')),
            'ca': int(form_data.get('ca')),
            'thal': int(form_data.get('thal'))
        }
        
        # Preprocess the input
        processed_input = preprocess_input(input_features)
        
        # Load the model and make prediction
        model, _ = get_model()  # Unpack model and scaler
        prediction = model.predict(processed_input)[0]
        probability = model.predict_proba(processed_input)[0][1]
        
        # Determine risk level based on probability
        if probability < 0.3:
            risk_level = "Low Risk"
            risk_class = "success"
        elif probability < 0.7:
            risk_level = "Moderate Risk"
            risk_class = "warning"
        else:
            risk_level = "High Risk"
            risk_class = "danger"
        
        # Calculate additional risk metrics (for visualization)
        age_factor = min(1.0, input_features['age'] / 100)
        chol_factor = min(1.0, input_features['chol'] / 300)
        bp_factor = min(1.0, input_features['trestbps'] / 180)
        
        return render_template(
            'result.html',
            prediction=prediction,
            probability=round(probability * 100, 2),
            risk_level=risk_level,
            risk_class=risk_class,
            patient_data=input_features,
            age_factor=round(age_factor * 100, 2),
            chol_factor=round(chol_factor * 100, 2),
            bp_factor=round(bp_factor * 100, 2)
        )
    
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction (for AJAX calls)"""
    try:
        data = request.json
        processed_input = preprocess_input(data)
        
        model, _ = get_model()  # Unpack model and scaler
        prediction = int(model.predict(processed_input)[0])
        probability = float(model.predict_proba(processed_input)[0][1])
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': round(probability * 100, 2),
            'risk_level': "High Risk" if probability >= 0.7 else "Moderate Risk" if probability >= 0.3 else "Low Risk"
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Make sure the model directory exists
    os.makedirs('model', exist_ok=True)
    
    # Run the Flask application
    app.run(debug=True)