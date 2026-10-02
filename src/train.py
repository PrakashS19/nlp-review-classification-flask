"""New runnable portfolio baseline, not the original coursework model.
Example: python src/train.py --input data/assignment3.csv
"""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from text_pipeline import normalize_review


def train(source: Path, out: Path):
    data = pd.read_csv(source)
    required = ['Review Text', 'Recommended IND']
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f'Missing dataset columns: {missing}')
    data = data.dropna(subset=['Recommended IND']).copy()
    data['Review Text'] = data['Review Text'].fillna('')
    data['clean_text'] = data['Review Text'].map(normalize_review)
    data = data[data['clean_text'].str.strip().ne('')].copy()
    data['Recommended IND'] = pd.to_numeric(data['Recommended IND'], errors='coerce')
    data = data.dropna(subset=['Recommended IND'])
    data = data[data['Recommended IND'].isin([0, 1])]
    if len(data) < 20 or data['Recommended IND'].value_counts().min() < 5:
        raise ValueError('Need at least 20 labelled reviews and at least five in each class')
    X_train, X_test, y_train, y_test = train_test_split(
        data['clean_text'], data['Recommended IND'].astype(int), test_size=0.2,
        stratify=data['Recommended IND'], random_state=42
    )
    model = Pipeline([
        ('vectorizer', CountVectorizer(lowercase=False, min_df=2, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=['accuracy', 'f1', 'roc_auc'])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        'scope': 'NEW portfolio baseline — NOT the recovered original coursework classifier',
        'label': 'Recommended IND (review recommendation, NOT sentiment or churn)',
        'samples': len(data), 'train_size': len(X_train), 'test_size': len(X_test),
        'cv_f1_mean': float(cv_results['test_f1'].mean()),
        'cv_roc_auc_mean': float(cv_results['test_roc_auc'].mean()),
        'test_roc_auc': float(roc_auc_score(y_test, probabilities)),
        'test_classification_report': classification_report(y_test, predictions, output_dict=True, zero_division=0)
    }
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / 'review_recommendation_model.joblib')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k:v for k,v in metrics.items() if k != 'test_classification_report'}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('data/womens_clothing_reviews.csv'))
    parser.add_argument('--out', type=Path, default=Path('artifacts'))
    args = parser.parse_args()
    train(args.input, args.out)
