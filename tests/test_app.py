import importlib.util
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pytest
pytest.importorskip("flask", reason="Install requirements.txt to run Flask integration tests")
from app import app

def test_reject_empty():
    c=app.test_client()
    r=c.post('/predict', json={'review': ''})
    assert r.status_code == 400

def test_missing_model_friendly_response(tmp_path, monkeypatch):
    import app as module
    monkeypatch.setattr(module, 'MODEL_PATH', tmp_path / 'absent.joblib')
    module.get_model.cache_clear()
    assert module.app.test_client().post('/predict', json={'review':'Good fabric'}).status_code == 503
