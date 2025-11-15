import numpy as np
import joblib
import os

# Load Model Files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  

# Path to project root
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Load scaler
scaler_path = os.path.join(ROOT_DIR, "src", "preprocessing", "scaler.pkl")
scaler = joblib.load(scaler_path)

# Load weights
weights_path = os.path.join(ROOT_DIR, "src", "models", "logistic_regression", "weights.npy")
bias_path = os.path.join(ROOT_DIR, "src", "models", "logistic_regression", "bias.npy")

w = np.load(weights_path)
b = np.load(bias_path)[0]


# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Prediction Function
def predict_customer(features):
    """
    features: list or array of length 20 (scaled by the scaler)
    """
    x = np.array(features).reshape(1, -1)
    x_scaled = scaler.transform(x)

    z = np.dot(x_scaled, w) + b
    p = sigmoid(z)[0]

    return p


# Risk category

def risk_category(p):
    if p < 0.3:
        return "Low Risk"
    elif p < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"


# Example Usage 

if __name__ == "__main__":
    # Example customer (raw values)
    example = [
        2,   # Status
        12,  # Duration
        2,   # CreditHistory
        3,   # Purpose
        2500, # Amount
        3,   # Savings
        2,   # Employment
        4,   # InstallRate
        1,   # PersonalStatus
        1,   # Debtors
        3,   # Residence
        4,   # Property
        45,  # Age
        2,   # InstallPlans
        1,   # Housing
        1,   # ExistingCredits
        3,   # Job
        1,   # Liable
        1,   # Telephone
        1    # Foreign
    ]

    p = predict_customer(example)
    print("Default probability:", p)
    print("Risk level:", risk_category(p))