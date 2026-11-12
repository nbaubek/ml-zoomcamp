"""Smoke test for the real, shipped model.bin (skipped when the file is absent)."""

import pytest
from fastapi.testclient import TestClient

from conftest import ROOT, load_predict_module

MODEL = ROOT / 'model.bin'


@pytest.mark.skipif(not MODEL.exists(), reason='model.bin not present')
def test_shipped_model_scores_the_readme_example_as_a_purchase(payload):
    client = TestClient(load_predict_module(MODEL, 'predict_shipped').app)
    body = client.post('/predict', json=payload).json()
    assert body['threshold'] == pytest.approx(0.15)
    assert body['purchase'] is True
    assert body['purchase_probability'] > 0.5