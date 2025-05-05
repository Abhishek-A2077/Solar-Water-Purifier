"""
Flask App: Water Purification Level Predictor

Requirements:
    pip install flask joblib numpy pandas scikit-learn packaging

Run:
    export FLASK_APP=app.py
    flask run
"""
from flask import Flask, request, render_template_string, redirect, url_for, flash
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'replace_with_secure_key'

# Verify scikit-learn version compatibility
try:
    import sklearn
    from packaging import version
    skl_ver = version.parse(sklearn.__version__)
    if not (skl_ver == version.parse("1.1.3") or skl_ver >= version.parse("1.2.2")):
        raise ImportError(f"Incompatible scikit-learn version {sklearn.__version__}. "
                          "Install scikit-learn==1.1.3 or >=1.2.2: pip install scikit-learn==1.1.3")
except Exception as e:
    raise RuntimeError(f"Dependency error: {e}")

# Load trained model
MODEL_PATH = "water_purifier_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
model = joblib.load(MODEL_PATH)

# HTML template
PAGE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Water Purification Predictor</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; }
    form { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; max-width: 800px; }
    label { font-weight: bold; }
    input, select { padding: 8px; border: 1px solid #ccc; border-radius: 4px; width: 100%; }
    .full { grid-column: span 2; text-align: center; margin-top: 20px; }
    .result { margin-top: 20px; font-size: 1.2em; color: green; }
    .flash { color: red; }
  </style>
</head>
<body>
  <h1>💧 Water Purification Predictor</h1>
  {% with messages = get_flashed_messages() %}
    {% if messages %}
      <ul class="flash">
      {% for msg in messages %}<li>{{ msg }}</li>{% endfor %}
      </ul>
    {% endif %}
  {% endwith %}
  <form method="post" action="/predict">
    {% for field in fields %}
      <div>
        <label>{{ field.label }}</label>
        <input type="number" name="{{ field.name }}" step="0.01" required value="{{ field.value }}">
      </div>
    {% endfor %}
    <div>
      <label>Filter Type</label>
      <select name="filter_type">
        {% for f in filters %}
          <option value="{{ f }}" {% if f==filters[0] %}selected{% endif %}>{{ f }}</option>
        {% endfor %}
      </select>
    </div>
    <div class="full"><button type="submit">Predict</button></div>
  </form>
  {% if purification is defined %}
    <div class="result">Predicted Purification: {{ purification }}%</div>
  {% endif %}
</body>
</html>
'''

# Field definitions
def get_fields():
    defaults = {
        'voltage': 18.0, 'current': 3.0, 'turbidity': 5.0, 'ph': 7.0,
        'tds': 300.0, 'conductivity': 450.0, 'water_temp': 25.0,
        'ambient_temp': 25.0, 'irradiance': 600.0, 'flow_rate': 1.2,
        'purification_time': 10
    }
    labels = {
        'voltage': 'Solar Voltage (V)', 'current': 'Solar Current (A)',
        'turbidity': 'Turbidity (NTU)', 'ph': 'pH Level', 'tds': 'TDS (ppm)',
        'conductivity': 'Conductivity (μS/cm)', 'water_temp': 'Water Temperature (°C)',
        'ambient_temp': 'Ambient Temperature (°C)', 'irradiance': 'Irradiance (W/m²)',
        'flow_rate': 'Flow Rate (L/min)', 'purification_time': 'Purification Time (min)'
    }
    return [{'name': k, 'label': labels[k], 'value': defaults[k]} for k in defaults]

@app.route('/', methods=['GET'])
def index():
    return render_template_string(PAGE, fields=get_fields(), filters=['RO','UV','Carbon','UV+Carbon'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {fld['name']: float(request.form[fld['name']]) for fld in get_fields()}
        filter_type = request.form['filter_type']
    except Exception:
        flash('Invalid input. Please enter numeric values.')
        return redirect(url_for('index'))

    # One-hot encode filter
    filter_list = ['Carbon','RO','UV','UV+Carbon']
    encoding = [1 if filter_type == f else 0 for f in filter_list]

    features = [
        data['voltage'], data['current'], data['turbidity'], data['ph'], data['tds'],
        data['conductivity'], data['water_temp'], data['ambient_temp'], data['irradiance'],
        data['flow_rate'], data['purification_time']
    ] + encoding

    # Predict purification level
    pred = model.predict(np.array(features).reshape(1, -1))[0]
    purification = round(pred, 2)

    # Log results
    log = pd.DataFrame([{**data, 'filter_type': filter_type, 'purification': purification}])
    log_file = 'predictions_log.csv'
    log.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

    return render_template_string(PAGE, fields=get_fields(), filters=filter_list, purification=purification)

if __name__ == '__main__':
    app.run(debug=True)
