import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from text_pipeline import normalize_review

def test_tokenization():
    assert normalize_review("WOW! Well-made, don't you think? 9/10") == "wow well-made don't you think"

def test_empty():
    assert normalize_review(None) == ''
