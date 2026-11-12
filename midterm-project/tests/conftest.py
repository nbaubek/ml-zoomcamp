"""Shared fixtures.

The API tests run against a small throwaway model trained on the committed dataset, so they
do not depend on model.bin and stay fast. One extra test (test_shipped_model.py) checks the
real model.bin when it is present.
"""

import importlib.util
import pickle
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier

import train

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / 'data' / 'online_shoppers_intention.csv'


def load_predict_module(model_path: Path, name: str):
    """Import predict.py as a separate module that reads the model from model_path."""
    spec = importlib.util.spec_from_file_location(name, ROOT / 'predict.py')
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('MODEL_PATH', str(model_path))
        spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='session')
def raw_df() -> pd.DataFrame:
    return pd.read_csv(DATA_CSV)


@pytest.fixture(scope='session')
def clean_df(raw_df) -> pd.DataFrame:
    return train.clean(raw_df.copy())


@pytest.fixture(scope='session')
def tiny_model_path(tmp_path_factory, clean_df) -> Path:
    """Train a small XGBoost model with the same preprocessing as train.py."""
    features = clean_df.drop(columns=['revenue'])
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(train.to_dicts(features))
    model = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1,
                          random_state=42, n_jobs=1, eval_metric='logloss')
    model.fit(X, clean_df['revenue'].values)

    path = tmp_path_factory.mktemp('model') / 'model.bin'
    with open(path, 'wb') as f:
        pickle.dump({'dv': dv, 'model': model, 'threshold': train.THRESHOLD}, f)
    return path


@pytest.fixture(scope='session')
def predict_module(tiny_model_path):
    return load_predict_module(tiny_model_path, 'predict_under_test')


@pytest.fixture(scope='session')
def client(predict_module) -> TestClient:
    return TestClient(predict_module.app)


@pytest.fixture
def payload() -> dict:
    """A valid request body (lowercase, standardized values, as the API expects)."""
    return {
        'administrative': 2, 'administrative_duration': 120.0,
        'informational': 1, 'informational_duration': 30.0,
        'productrelated': 40, 'productrelated_duration': 1800.0,
        'bouncerates': 0.0, 'exitrates': 0.02,
        'pagevalues': 25.0, 'specialday': 0.0,
        'month': 'nov', 'operatingsystems': 2, 'browser': 2,
        'region': 1, 'traffictype': 2,
        'visitortype': 'new_visitor', 'weekend': False,
    }