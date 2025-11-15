# Heart Disease Prediction System - Setup Instructions

## Prerequisites
- Python 3.8 or higher
- Virtual environment (venv) - already set up in this project

## Setup Steps

### 1. Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```

**On Linux/Mac:**
```bash
source venv/bin/activate
```

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

### 3. Prepare Your Dataset

Make sure you have your heart disease dataset CSV file. The app will look for:
- `heart (1).csv` or
- `heart.csv`

Place the dataset file in the root directory of the project.

### 4. Run the Application

```bash
python app.py
```

The application will:
- First try to load a saved model (`heart_disease_model.pkl`)
- If no saved model exists, it will train a new one from your dataset
- Start the Flask server on `http://127.0.0.1:5000`

### 5. Access the Web Interface

Open your web browser and navigate to:
```
http://localhost:5000
```
or
```
http://127.0.0.1:5000
```

## Quick Start Commands (Windows PowerShell)

```powershell
# Navigate to project directory
cd "C:\Users\gupta\OneDrive\Desktop\Heart-Disease-Prediction-Using-RandomForest"

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## Troubleshooting

### If you get "Model not loaded" error:
- Make sure your dataset CSV file is in the root directory
- Check that the CSV file has the correct column names matching the form fields
- The app will automatically train a model on first run

### If port 5000 is already in use:
- Change the port in `app.py` (last line): `app.run(debug=True, host='0.0.0.0', port=5001)`

### If you encounter import errors:
- Make sure the virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt --upgrade`

## Notes

- The model will be saved as `heart_disease_model.pkl` after first training
- Subsequent runs will load the saved model (much faster)
- To retrain the model, delete `heart_disease_model.pkl` and restart the app

