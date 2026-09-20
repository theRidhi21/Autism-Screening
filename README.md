# Autism Spectrum Screening

A web app that runs the **AQ-10 screening questionnaire** through a trained
classifier and returns a screening result.

**Live demo:** _add your PythonAnywhere URL here after deploying_

---

## What it does

The user answers the 10 AQ-10 questions plus a few demographic fields. The app
scores the questionnaire, builds the 94-feature vector the model expects, and
returns whether the answers show indicators of autism spectrum traits.

The AQ-10 reverse-scores 6 of its 10 items, so the app maps each Agree/Disagree
answer to the correct point value rather than taking answers at face value.

## Model

Seven classifiers were trained and compared (see `ML_models.ipynb`). The
deployed model is the linear **SVM**:

| Model | Test Accuracy | Test F1 | Precision | Recall | ROC AUC |
|---|---|---|---|---|---|
| Random Forest | 0.941 | 0.946 | 0.945 | 0.948 | 0.990 |
| Decision Tree | 0.927 | 0.934 | 0.937 | 0.930 | 0.927 |
| kNN | 0.882 | 0.892 | 0.898 | 0.885 | 0.945 |
| Logistic Regression | 0.862 | 0.875 | 0.867 | 0.883 | 0.952 |
| **SVM (deployed)** | **0.833** | **0.835** | **0.909** | **0.773** | **0.942** |
| LDA | 0.827 | 0.834 | 0.881 | 0.793 | 0.945 |
| Naive Bayes | 0.821 | 0.823 | 0.894 | 0.763 | 0.896 |

### A note on what the model actually learned

Holding the AQ-10 score fixed and varying every other input — age, gender,
jaundice at birth, family history, ethnicity, country, relation — does not
change the prediction. In practice the model is a threshold on the
questionnaire score (it flips to positive at 7 of 10), and the 88 demographic
one-hot features contribute almost nothing.

This is a property of the dataset rather than a flaw in training: the
`Class/ASD` label is itself derived from the questionnaire total, so the model
is largely rediscovering the AQ-10's own cutoff.

## Tech stack

Flask · scikit-learn · pandas · joblib · gunicorn · vanilla JS frontend

## Running locally

```bash
pip install -r requirements.txt
python app.py              # http://127.0.0.1:5000
FLASK_DEBUG=1 python app.py  # with reloader
```

## Dataset

Autism Screening Adult dataset (Fadi Fayez Thabtah), UCI Machine Learning
Repository.

## Disclaimer

This is a student project, not a medical device. The AQ-10 is a screening
questionnaire, not a diagnostic instrument, and this model misses roughly a
quarter of true positives (recall 0.77). It cannot diagnose autism — only a
qualified clinician can.
