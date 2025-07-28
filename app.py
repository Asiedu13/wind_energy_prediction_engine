from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import numpy as np
import pandas as pd
import joblib
from model import PowerRNN

app = Flask(__name__)
CORS(app) 
# Load model and scaler
model = PowerRNN()
model.load_state_dict(torch.load("./models/best_wind_power_rnn.pth", map_location=torch.device('cpu')))
model.eval()
scaler = joblib.load("./models/wind_power_scaler.save")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def predict():
    file = request.files['file']
    df = pd.read_csv(file)

    # Make a copy to preserve original structure with 'Time' if needed
    original_df = df.copy()
    print(df.head())
    print(df.shape)
    # Sort by datetime if time column exists
    time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    if time_cols:
        print('editing time')
        df[time_cols[0]] = pd.to_datetime(df[time_cols[0]])
        df = df.sort_values(by=time_cols[0])
        df[time_cols[0]] = 0

    # Drop only the time/date column from numeric input (if present)
    numeric_df = df.select_dtypes(include=[float, int])
    print('finished type selection')
    print(df.head())
    print(df.shape)
    
    # If only 8 numeric features, try dropping the time column explicitly
    # if numeric_df.shape[1] == 9 and time_cols:
        # df_numeric_full = df.drop(columns=time_cols)
        # print('df_numeric_full\n', df_numeric_full)
        
        # numeric_df = df_numeric_full.select_dtypes(include=[float, int])

    # Final validation
    if numeric_df.shape[1] != 9:
        return jsonify({"error": f"Expected 9 numeric columns, got {numeric_df.shape[1]}"}), 400

    print('last last')
    print(numeric_df.head())
    print(numeric_df.shape)
    
    # Scale and predict
    input_scaled = scaler.transform(numeric_df.values)
    input_tensor = torch.tensor([input_scaled], dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor).item()

    dummy = np.zeros((1, scaler.n_features_in_))
    dummy[0, -1] = output
    power = scaler.inverse_transform(dummy)[0, -1]

    return jsonify({"predicted_power": round(power, 3)})

if __name__ == "__main__":
    app.run(debug=True)
