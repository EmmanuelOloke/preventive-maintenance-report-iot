from flask import Flask, render_template, jsonify, request
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

app = Flask(__name__)

# Sample parameters and their acceptable ranges
system_parameters = [
    {"parameter": "Water production meter", "range": (900, 1100), "unit": "gpd"},
    {"parameter": "Doping unit level/water solution tank", "range": (7.5, 8.5), "unit": "litres"},
    {"parameter": "Chemical feed pump value", "range": (7, 9), "unit": "bar"},
    {"parameter": "Chlorine residue value at application point", "range": (0, 3.5), "unit": "ppm"},
    {"parameter": "Chlorine residue value at distribution point", "range": (0, 3.5), "unit": "ppm"},
    {"parameter": "Booster pump value", "range": (75, 100), "unit": "psi"},
    {"parameter": "Chlorine concentration in distribution system", "range": (0, 3.5), "unit": "ppm"},
    {"parameter": "Carbonization effect in filtration system", "range": (75, 100), "unit": "psi"},
    {"parameter": "UV purifier discharge flow rate", "range": (90, 100), "unit": "psi"},
    {"parameter": "Neutralizing tank", "range": (75, 75), "unit": "psi"},
    {"parameter": "Back flush system", "range": (0, 0), "unit": "psi"},
    {"parameter": "Chemical feed tank", "range": (60, 80), "unit": "litres"},
    {"parameter": "Product water valve", "range": (0, 8), "unit": "psi"},
    {"parameter": "High pressure pump value", "range": (250, 250), "unit": "psi"},
    {"parameter": "Carbon filters", "range": (0, 8), "unit": "psi"},
    {"parameter": "Cross flow", "range": (0.8, 0.9), "unit": "gpm"},
    {"parameter": "Chlorination and De-chlorination dosing unit", "range": (0.4, 0.6), "unit": "ppm"},
    {"parameter": "Antiscalant dosing tank level", "range": (75, 80), "unit": "litres"},
    {"parameter": "Second filtration cartridge filters", "range": (7, 8), "unit": "psi"},
    {"parameter": "Feeding water / high TDS raw water", "range": (50, 150), "unit": "ppm"},
    {"parameter": "Sodium hydroxide PH value", "range": (0.6, 0.8), "unit": "ppm"}
]

# Function to generate random data for each parameter
def generate_random_data():
    random_data = []
    for param in system_parameters:
        value = round(random.uniform(*param["range"]), 2)  # Generate random value within range
        random_data.append({
            "parameter": param["parameter"],
            "value": value,
            "unit": param["unit"]
        })
    return random_data

# Function to generate a dataset and train a Random Forest model
def generate_dataset_and_train_model(num_samples=1000):
    data = []
    statuses = ['G', 'Y', 'R', 'LO', 'RO', 'PW', 'BW', 'RN', 'PH', 'ST']
    
    for _ in range(num_samples):
        sample = generate_random_data()
        status = random.choice(statuses)
        data_row = {param["parameter"]: item["value"] for param, item in zip(system_parameters, sample)}
        data_row['status'] = status
        data.append(data_row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Prepare data for Random Forest model
    X = df.drop('status', axis=1)  # Features
    y = df['status']  # Target

    # Convert categorical target to numerical encoding
    y = pd.factorize(y)[0]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=sorted(set(y)))

    return df, model, report


@app.route('/')
def simulation():
    # Load the dashboard template with system parameters and statuses
    return render_template('simulation.html')

@app.route('/simulation/start', methods=['POST'])
def start_simulation():
    random_data = generate_random_data()  # Generate random data
    status = random.choice(['G', 'Y', 'R', 'LO', 'RO', 'PW', 'BW', 'RN', 'PH', 'ST'])
    return jsonify({"message": "Simulation started!", "data": random_data, "status": status})

@app.route('/simulation/stop', methods=['POST'])
def stop_simulation():
    return jsonify({"message": "Simulation stopped!"})

@app.route('/train_model', methods=['POST'])
def train_model():
    df, model, report = generate_dataset_and_train_model()
    return jsonify({"message": "Model trained successfully!", "report": report})

if __name__ == '__main__':
    app.run(debug=True)
