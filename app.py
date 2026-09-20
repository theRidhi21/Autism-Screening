import os

from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)

# Load the model. The path is absolute because the server's working directory
# is not necessarily this folder (PythonAnywhere runs from elsewhere).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
classifier = joblib.load(os.path.join(BASE_DIR, 'saved_model.pkl'))

# Define your model columns based on your previous setup
model_columns=['age',
 'result',
 'A1_Score',
 'A2_Score',
 'A3_Score',
 'A4_Score',
 'A5_Score',
 'A6_Score',
 'A7_Score',
 'A8_Score',
 'A9_Score',
 'A10_Score',
 'gender_f',
 'gender_m',
 'ethnicity_Asian',
 'ethnicity_Black',
 'ethnicity_Hispanic',
 'ethnicity_Latino',
 'ethnicity_Middle Eastern ',
 'ethnicity_Others',
 'ethnicity_Pasifika',
 'ethnicity_South Asian',
 'ethnicity_Turkish',
 'ethnicity_White-European',
 'ethnicity_others',
 'jundice_no',
 'jundice_yes',
 'austim_no',
 'austim_yes',
 'contry_of_res_Afghanistan',
 'contry_of_res_AmericanSamoa',
 'contry_of_res_Angola',
 'contry_of_res_Armenia',
 'contry_of_res_Aruba',
 'contry_of_res_Australia',
 'contry_of_res_Austria',
 'contry_of_res_Bahamas',
 'contry_of_res_Bangladesh',
 'contry_of_res_Belgium',
 'contry_of_res_Bolivia',
 'contry_of_res_Brazil',
 'contry_of_res_Burundi',
 'contry_of_res_Canada',
 'contry_of_res_Chile',
 'contry_of_res_China',
 'contry_of_res_Costa Rica',
 'contry_of_res_Cyprus',
 'contry_of_res_Czech Republic',
 'contry_of_res_Ecuador',
 'contry_of_res_Egypt',
 'contry_of_res_Ethiopia',
 'contry_of_res_Finland',
 'contry_of_res_France',
 'contry_of_res_Germany',
 'contry_of_res_Iceland',
 'contry_of_res_India',
 'contry_of_res_Indonesia',
 'contry_of_res_Iran',
 'contry_of_res_Ireland',
 'contry_of_res_Italy',
 'contry_of_res_Jordan',
 'contry_of_res_Malaysia',
 'contry_of_res_Mexico',
 'contry_of_res_Nepal',
 'contry_of_res_Netherlands',
 'contry_of_res_New Zealand',
 'contry_of_res_Nicaragua',
 'contry_of_res_Niger',
 'contry_of_res_Oman',
 'contry_of_res_Pakistan',
 'contry_of_res_Philippines',
 'contry_of_res_Portugal',
 'contry_of_res_Romania',
 'contry_of_res_Russia',
 'contry_of_res_Saudi Arabia',
 'contry_of_res_Serbia',
 'contry_of_res_Sierra Leone',
 'contry_of_res_South Africa',
 'contry_of_res_Spain',
 'contry_of_res_Sri Lanka',
 'contry_of_res_Sweden',
 'contry_of_res_Tonga',
 'contry_of_res_Turkey',
 'contry_of_res_Ukraine',
 'contry_of_res_United Arab Emirates',
 'contry_of_res_United Kingdom',
 'contry_of_res_United States',
 'contry_of_res_Uruguay',
 'contry_of_res_Viet Nam',
 'relation_Health care professional',
 'relation_Others',
 'relation_Parent',
 'relation_Relative',
 'relation_Self']


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
        processed_data = preprocess_data(form_data)
    except (KeyError, ValueError) as exc:
        return jsonify({'error': f'Invalid input: {exc}'}), 400

    # Make prediction
    prediction = classifier.predict(processed_data)
    result = "ASD" if prediction[0] == 1 else "No ASD"

    return jsonify({'result': result,
                    'score': int(processed_data.loc[0, 'result'])})

def preprocess_data(form_data):
    # Initialize data for all model features
    data = pd.DataFrame(0, columns=model_columns, index=[0], dtype=float)

    # Numeric data
    age = float(form_data['age'])
    if not 1 <= age <= 120:
        raise ValueError('age must be between 1 and 120')
    data.loc[0, 'age'] = age

    # Scores and result
    for i in range(1, 11):
        score = int(form_data.get(f'a{i}_score', 0))
        if score not in (0, 1):
            raise ValueError(f'a{i}_score must be 0 or 1')
        data.loc[0, f'A{i}_Score'] = score

    # Calculate result based on A1_Score to A10_Score
    data.loc[0, 'result'] = sum(data.loc[0, f'A{i}_Score'] for i in range(1, 11))

    # Gender
    if form_data['gender'] == 'f':
        data.loc[0, 'gender_f'] = 1
    else:
        data.loc[0, 'gender_m'] = 1

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
    if column not in data.columns:
        raise ValueError(f'unknown value {value!r} for {prefix.rstrip("_")}')
    data.loc[0, column] = 1

def set_yes_no(data, prefix, value):
    if value not in ('yes', 'no'):
        raise ValueError(f'{prefix} must be yes or no')
    data.loc[0, f'{prefix}_{value}'] = 1

if __name__ == '__main__':
    # Debug must stay off in production: the Werkzeug debugger allows arbitrary
    # code execution. Set FLASK_DEBUG=1 locally when you want the reloader.
    app.run(host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)),
            debug=os.environ.get('FLASK_DEBUG') == '1')
