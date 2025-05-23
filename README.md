# Heart Disease Detection System

A machine learning application that predicts heart disease risk based on patient data.

![Heart Disease Detection](https://img.shields.io/badge/Health-AI-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-2.0+-lightgrey)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange)

## Overview

This application uses a machine learning model to predict heart disease risk based on patient data. The system provides an intuitive web interface for healthcare professionals to input patient information and receive instant risk assessments with visual feedback.

## How It Works

### Backend (Flask)
- The application uses Flask as the web framework to handle HTTP requests and serve HTML pages
- The main application file (`app.py`) defines routes for the homepage and prediction endpoint
- When the application starts, it initializes a machine learning model or loads a pre-trained one

### Machine Learning Model
- The model in `heart_model.py` is a RandomForestClassifier from scikit-learn
- For demonstration purposes, it generates synthetic training data with realistic distributions
- The model is trained to predict heart disease risk based on 13 clinical features
- Trained model is saved to disk and loaded when needed for predictions

### Frontend
- The user interface is built with HTML, CSS, and JavaScript, using Bootstrap for responsive design
- The main page (`index.html`) contains a form for users to input patient data
- The results page (`result.html`) displays the prediction with visualizations using Chart.js

### User Flow
1. User enters patient data through the form on the homepage
2. On submission, the data is sent to the server for processing
3. The server preprocesses the data, makes a prediction using the machine learning model, and returns the results
4. The results are displayed with visual cues (color-coded risk levels, charts) and recommendations

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions
1. Clone the repository:
   ```
   git clone https://github.com/shivamv97/Heart_disease_predict.git
   cd heart-disease-detection
   ```

2. Set up a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python app.py
   ```

5. Open a web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## Project Structure

```
heart-disease-detection/
├── app.py                    # Main Flask application
├── heart_model.py            # Machine learning model implementation
├── requirements.txt          # Python dependencies
├── static/
│   ├── css/                  # Stylesheet files
│   ├── js/                   # JavaScript files
│   └── img/                  # Image assets
└── templates/
    ├── index.html            # Homepage with input form
    └── result.html           # Results page with visualizations
```

## Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, NumPy, Pandas
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap
- **Visualization**: Chart.js

## Future Enhancements

- Integration with electronic health record (EHR) systems
- Additional machine learning models for comparison
- More detailed patient recommendations
- User authentication and patient data management
- Mobile application version
