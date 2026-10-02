"""New minimal Flask inference demonstration for the reconstructed classifier."""
import os
from pathlib import Path
from functools import lru_cache
import joblib
from flask import Flask, jsonify, request
from text_pipeline import normalize_review

app = Flask(__name__)
MODEL_PATH = Path(os.environ.get('MODEL_PATH', 'artifacts/review_recommendation_model.joblib'))

@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError('Train the model first: python src/train.py --input data/womens_clothing_reviews.csv')
    return joblib.load(MODEL_PATH)  # Load only a trusted model you trained locally.

@app.get('/')
def home():
    return '<h2>Review recommendation demo</h2><p>POST JSON {"review": "your review"} to /predict.</p>'

@app.post('/predict')
def predict():
    payload = request.get_json(silent=True) or {}
    review = payload.get('review', '')
    if not isinstance(review, str) or not review.strip() or len(review) > 5000:
        return jsonify({'error': 'review must be nonempty text (max 5000 characters)'}), 400
    try:
        model = get_model()
    except FileNotFoundError as error:
        return jsonify({'error': str(error)}), 503
    clean = normalize_review(review)
    probability = float(model.predict_proba([clean])[0, 1])
    return jsonify({'recommendation_probability': round(probability, 4), 'predicted_recommended': bool(probability >= 0.5), 'notice': 'Review recommendation prediction, not a factual product endorsement.'})

if __name__ == '__main__':
    app.run(debug=False)
