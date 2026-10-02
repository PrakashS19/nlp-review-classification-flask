from pathlib import Path
import sys, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from train import train

def test_smoke_training(tmp_path):
    # Synthetic data tests software flow only, NOT project model accuracy.
    texts=['beautiful comfortable elegant fit %s'%i for i in range(25)]+['poor itchy disappointing broken %s'%i for i in range(25)]
    labels=[1]*25+[0]*25
    p=tmp_path/'synthetic.csv';pd.DataFrame({'Review Text':texts,'Recommended IND':labels}).to_csv(p,index=False)
    train(p,tmp_path/'out')
    assert (tmp_path/'out'/'review_recommendation_model.joblib').exists()
    assert (tmp_path/'out'/'metrics.json').exists()
