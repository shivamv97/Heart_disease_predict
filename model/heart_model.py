import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

class HeartDiseaseModel:
    """
    A class for training and using a heart disease prediction model.
    Uses a RandomForestClassifier from scikit-learn.
    """
    
    def __init__(self):
        """Initialize the model and scaler"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.features = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
    
    def _generate_sample_data(self):
        """
        Generate sample heart disease data for training.
        This is a simplified version with synthetic data.
        In a real application, you would use actual medical data.
        """
        np.random.seed(42)
        n_samples = 500
        
        # Generate features with realistic distributions
        data = {
            'age': np.random.normal(55, 10, n_samples).astype(int),
            'sex': np.random.binomial(1, 0.7, n_samples),  # 0: female, 1: male
            'cp': np.random.randint(0, 4, n_samples),  # Chest pain type (0-3)
            'trestbps': np.random.normal(130, 20, n_samples).astype(int),  # Resting blood pressure
            'chol': np.random.normal(230, 40, n_samples).astype(int),  # Cholesterol
            'fbs': np.random.binomial(1, 0.15, n_samples),  # Fasting blood sugar > 120 mg/dl
            'restecg': np.random.randint(0, 3, n_samples),  # Resting ECG results
            'thalach': np.random.normal(150, 20, n_samples).astype(int),  # Max heart rate
            'exang': np.random.binomial(1, 0.3, n_samples),  # Exercise induced angina
            'oldpeak': np.random.normal(1.0, 1.0, n_samples).round(1),  # ST depression induced by exercise
            'slope': np.random.randint(0, 3, n_samples),  # Slope of the peak exercise ST segment
            'ca': np.random.multinomial(3, [0.7, 0.15, 0.1, 0.05], n_samples).argmax(axis=1),  # Number of major vessels
            'thal': np.random.randint(0, 3, n_samples)  # Thalassemia (0: normal, 1: fixed defect, 2: reversible defect)
        }
        
        # Generate target variable based on features (simplified model)
        # Higher age, male sex, abnormal ECG, high BP, high cholesterol increase risk
        risk_score = (
            0.05 * data['age'] +
            0.3 * data['sex'] + 
            0.2 * data['cp'] + 
            0.01 * data['trestbps'] +
            0.003 * data['chol'] +
            0.2 * data['fbs'] + 
            0.2 * data['exang'] +
            0.1 * data['oldpeak'] +
            0.5 * data['ca'] +
            0.3 * data['thal'] -
            0.01 * data['thalach'] +
            np.random.normal(0, 1, n_samples)
        )
        
        # Convert to binary outcome
        data['target'] = (risk_score > np.median(risk_score)).astype(int)
        
        # Ensure values are within realistic ranges
        data['age'] = np.clip(data['age'], 20, 90)
        data['trestbps'] = np.clip(data['trestbps'], 90, 200)
        data['chol'] = np.clip(data['chol'], 120, 400)
        data['thalach'] = np.clip(data['thalach'], 60, 220)
        data['oldpeak'] = np.clip(data['oldpeak'], 0, 6)
        
        return pd.DataFrame(data)
    
    def train(self):
        """Train the model on sample data"""
        # In a real application, load actual data from a CSV or database
        data = self._generate_sample_data()
        
        # Split features and target
        X = data[self.features]
        y = data['target']
        
        # Split training and testing data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        accuracy = self.model.score(X_test_scaled, y_test)
        print(f"Model accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def predict(self, X):
        """Make predictions on new data"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability estimates for predictions"""
        return self.model.predict_proba(X)
    
    def save_model(self, filepath):
        """Save the model and scaler to disk"""
        # Save as a tuple of (model, scaler)
        joblib.dump((self.model, self.scaler), filepath)


def preprocess_input(input_data):
    """
    Preprocess input data for prediction.
    
    Args:
        input_data: Dictionary containing patient data
        
    Returns:
        Numpy array of processed features ready for prediction
    """
    # Define the expected feature order
    features = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    
    # Create a numpy array from the input data
    input_array = np.array([[
        input_data.get('age', 55),
        input_data.get('sex', 1),
        input_data.get('cp', 0),
        input_data.get('trestbps', 130),
        input_data.get('chol', 230),
        input_data.get('fbs', 0),
        input_data.get('restecg', 0),
        input_data.get('thalach', 150),
        input_data.get('exang', 0),
        input_data.get('oldpeak', 1.0),
        input_data.get('slope', 1),
        input_data.get('ca', 0),
        input_data.get('thal', 1)
    ]])
    
    try:
        # Load the saved model and scaler
        model_path = os.path.join('model', 'model.pkl')
        if os.path.exists(model_path):
            _, scaler = joblib.load(model_path)
            # Scale the input
            scaled_input = scaler.transform(input_array)
            return scaled_input
        else:
            print("Warning: Model file not found, returning unscaled input")
            return input_array
    except Exception as e:
        print(f"Error loading model/scaler: {e}")
        return input_array


def load_model():
    """Load the trained model and scaler from disk"""
    model_path = os.path.join('model', 'model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")