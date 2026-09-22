import json
import os

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)

# The trained SVM is linear, so its decision function is just a dot product
# plus an intercept -- no need to ship scikit-learn/pandas/numpy just to run
# it. The weights below were extracted once from saved_model.pkl (see
# extract_weights.py) and are loaded here as plain JSON. The path is absolute
# because the server's working directory is not necessarily this folder
# (PythonAnywhere runs from elsewhere).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, 'model_weights.json')) as f:
    _weights = json.load(f)

model_columns = _weights['features']
_coef = _weights['coef']
_intercept = _weights['intercept']
_classes = _weights['classes']


def predict_asd(feature_dict):
    """Binary linear-SVM decision rule: score >= 0 picks the second (higher)
    class, matching scikit-learn's convention for a two-class SVC."""
    score = _intercept
    for name, weight in zip(model_columns, _coef):
        score += weight * feature_dict[name]
    return _classes[1] if score >= 0 else _classes[0]


def options_for(prefix):
    """The exact category values the model was trained on, taken from the
    column names themselves so the form can never drift out of sync."""
    return [c[len(prefix):] for c in model_columns if c.startswith(prefix)]


ETHNICITIES = options_for('ethnicity_')
COUNTRIES = options_for('contry_of_res_')
RELATIONS = options_for('relation_')


@app.route('/')
def home():
    return render_template('index.html',
                           ethnicities=ETHNICITIES,
                           countries=COUNTRIES,
                           relations=RELATIONS)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract and convert form data to dictionary
    form_data = request.form.to_dict()
    try:
        features = preprocess_data(form_data)
    except (KeyError, ValueError) as exc:
        return jsonify({'error': f'Invalid input: {exc}'}), 400

    # Make prediction
    prediction = predict_asd(features)
    result = "ASD" if prediction == 1 else "No ASD"

    return jsonify({'result': result,
                    'score': int(features['result'])})

def preprocess_data(form_data):
    # Initialize every model feature to 0
    data = {name: 0.0 for name in model_columns}

    # Numeric data
    age = float(form_data['age'])
    if not 1 <= age <= 120:
        raise ValueError('age must be between 1 and 120')
    data['age'] = age

    # Scores and result
    for i in range(1, 11):
        score = int(form_data.get(f'a{i}_score', 0))
        if score not in (0, 1):
            raise ValueError(f'a{i}_score must be 0 or 1')
        data[f'A{i}_Score'] = score

    # Calculate result based on A1_Score to A10_Score
    data['result'] = sum(data[f'A{i}_Score'] for i in range(1, 11))

    # Gender
    if form_data['gender'] == 'f':
        data['gender_f'] = 1
    else:
        data['gender_m'] = 1

    # Ethnicity, country and relation are one-hot columns whose names contain
    # spaces, so the form value is used verbatim -- no substitution.
    set_one_hot(data, 'ethnicity_', form_data['ethnicity'])
    set_one_hot(data, 'contry_of_res_', form_data['country_of_res'])
    set_one_hot(data, 'relation_', form_data['relation'])

    # Jaundice and autism in the family: both sides of the pair are one-hot
    # columns, so the "no" column has to be set too.
    set_yes_no(data, 'jundice', form_data['jaundice'])
    set_yes_no(data, 'austim', form_data['austim'])

    return data

def set_one_hot(data, prefix, value):
    column = prefix + value
    if column not in data:
        raise ValueError(f'unknown value {value!r} for {prefix.rstrip("_")}')
    data[column] = 1

def set_yes_no(data, prefix, value):
    if value not in ('yes', 'no'):
        raise ValueError(f'{prefix} must be yes or no')
    data[f'{prefix}_{value}'] = 1

if __name__ == '__main__':
    # Debug must stay off in production: the Werkzeug debugger allows arbitrary
    # code execution. Set FLASK_DEBUG=1 locally when you want the reloader.
    app.run(host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)),
            debug=os.environ.get('FLASK_DEBUG') == '1')
