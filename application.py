import pickle
from flask import Flask, request, render_template, flash
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize the Flask app
application = Flask(__name__)
app = application
app.secret_key = "your_secret_key"  # Needed for flash messages

# Load the model and scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Home route
@app.route("/")
def index():
    return render_template("index.html")

# Prediction route
@app.route("/predictdata", methods=["POST"])
def predict_datapoint():
    try:
        # Get data from form
        data = [
            float(request.form["Temperature"]),
            float(request.form["RH"]),
            float(request.form["Ws"]),
            float(request.form["Rain"]),
            float(request.form["FFMC"]),
            float(request.form["DMC"]),
            float(request.form["ISI"]),
            float(request.form["Classes"]),
            float(request.form["Region"]),
        ]
        
        # Scale the input
        scaled_data = standard_scaler.transform([data])
        prediction = ridge_model.predict(scaled_data)[0]
        
        return render_template("result.html", prediction=round(prediction, 2))
    
    except Exception as e:
        flash(f"Error: {str(e)}. Please check your inputs.")
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
