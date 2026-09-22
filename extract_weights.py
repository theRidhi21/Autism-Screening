"""One-off script: extracts the trained linear SVM's weights from
saved_model.pkl into model_weights.json.

Run this again only if the model is retrained. It requires scikit-learn and
joblib, which the deployed app itself does not: app.py reads model_weights.json
directly and does the dot product in plain Python, so the free-tier deploy
never needs to install scikit-learn/pandas/numpy/scipy.

    python3 extract_weights.py
"""
import json

import joblib

MODEL_PATH = 'saved_model.pkl'
OUTPUT_PATH = 'model_weights.json'


def main():
    model = joblib.load(MODEL_PATH)
    if model.kernel != 'linear':
        raise SystemExit(
            f"Model kernel is {model.kernel!r}, not 'linear' -- this script "
            "only supports extracting a linear SVM's weights for a plain dot "
            "product. A different kernel needs the real scikit-learn model "
            "at prediction time."
        )

    weights = {
        'features': list(model.feature_names_in_),
        'coef': [float(c) for c in model.coef_[0]],
        'intercept': float(model.intercept_[0]),
        'classes': [int(c) for c in model.classes_],
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(weights, f, indent=1)
    print(f'Wrote {OUTPUT_PATH} ({len(weights["features"])} features)')


if __name__ == '__main__':
    main()
