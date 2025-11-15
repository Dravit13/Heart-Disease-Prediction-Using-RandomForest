from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

app = Flask(__name__)

# Global variable to store the model
model = None
MODEL_FILE = 'heart_disease_model.pkl'
DATA_FILE = 'cardio_train.csv'

def load_or_train_model():
    """Load saved model or train a new one"""
    global model
    
    # Try to load saved model
    if os.path.exists(MODEL_FILE):
        print(f"Loading saved model from {MODEL_FILE}...")
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
        return
    
    # If no saved model, train a new one
    print("No saved model found. Training new model...")
    print("Note: This requires the heart dataset CSV file.")
    print("Please ensure you have the dataset file available.")
    
    # Try to load the dataset
    try:
        # Try to load heart dataset (adjust path as needed)
        if os.path.exists('heart (1).csv'):
            df = pd.read_csv('heart (1).csv')
        elif os.path.exists('heart.csv'):
            df = pd.read_csv('heart.csv')
        else:
            print("ERROR: Heart dataset not found!")
            print("Please provide the heart dataset CSV file.")
            return
        
        # Preprocess the data
        df_encoded = pd.get_dummies(df, drop_first=True)
        
        # Prepare features and target
        X = df_encoded.drop('HeartDisease', axis=1)
        y = df_encoded['HeartDisease']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Save model
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        
        # Save feature names for reference
        with open('feature_names.pkl', 'wb') as f:
            pickle.dump(list(X.columns), f)
        
        accuracy = model.score(X_test, y_test)
        print(f"Model trained successfully! Accuracy: {accuracy:.2f}")
        print(f"Model saved to {MODEL_FILE}")
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        print("The app will still run, but predictions won't work until a model is available.")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    global model
    
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please ensure the model is trained and available.'
        }), 500
    
    try:
        # Get data from request
        data = request.json
        
        # Create DataFrame with the input data
        input_data = pd.DataFrame([{
            'Age': data['Age'],
            'Sex': data['Sex'],
            'ChestPainType': data['ChestPainType'],
            'RestingBP': data['RestingBP'],
            'Cholesterol': data['Cholesterol'],
            'FastingBS': data['FastingBS'],
            'RestingECG': data['RestingECG'],
            'MaxHR': data['MaxHR'],
            'ExerciseAngina': data['ExerciseAngina'],
            'Oldpeak': data['Oldpeak'],
            'ST_Slope': data['ST_Slope']
        }])
        
        # One-hot encode categorical variables
        input_encoded = pd.get_dummies(input_data)
        
        # Load feature names to ensure correct order
        try:
            with open('feature_names.pkl', 'rb') as f:
                feature_names = pickle.load(f)
            
            # Reindex to match training features
            for col in feature_names:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            input_encoded = input_encoded[feature_names]
        except:
            # If feature names file doesn't exist, use current columns
            pass
        
        # Make prediction
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0]
        
        # Get probability of positive class (heart disease)
        if len(probability) > 1:
            prob_positive = probability[1]
        else:
            prob_positive = probability[0] if prediction == 1 else 1 - probability[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(prob_positive)
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 400

if __name__ == '__main__':
    # Load or train model on startup
    load_or_train_model()
    
    # Run the app
    print("\n" + "="*50)
    print("Heart Disease Prediction System")
    print("="*50)
    print("Server starting...")
    print("Open your browser and navigate to: http://127.0.0.1:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

