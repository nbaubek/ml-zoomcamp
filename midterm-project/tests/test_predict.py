"""Tests for the FastAPI service in predict.py."""

import pytest

import train


def test_health(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_predict_returns_valid_prediction(client, payload):
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {'purchase_probability', 'purchase', 'threshold'}
    assert 0.0 <= body['purchase_probability'] <= 1.0
    assert isinstance(body['purchase'], bool)
    assert body['threshold'] == pytest.approx(0.15)


def test_high_pagevalues_scores_higher_than_zero(client, payload):
    low = client.post('/predict', json={**payload, 'pagevalues': 0.0}).json()
    high = client.post('/predict', json={**payload, 'pagevalues': 40.0}).json()
    assert high['purchase_probability'] > low['purchase_probability']


def test_decision_follows_threshold(client, payload):
    clear_no = client.post('/predict', json={
        **payload, 'pagevalues': 0.0, 'exitrates': 0.2, 'productrelated': 1,
        'productrelated_duration': 0.0, 'administrative': 0,
        'administrative_duration': 0.0, 'month': 'feb'}).json()
    clear_yes = client.post('/predict', json={**payload, 'pagevalues': 40.0}).json()
    assert clear_no['purchase'] is False
    assert clear_yes['purchase'] is True


@pytest.mark.parametrize('field, bad_value', [
    ('month', 'Nov'),                        # raw-CSV casing is not accepted
    ('month', 'jan'),                        # month that never occurs in the data
    ('visitortype', 'Returning_Visitor'),    # raw-CSV casing
    ('bouncerates', 3),                      # rate above 1
    ('exitrates', -0.1),                     # negative rate
    ('administrative_duration', -5),         # negative duration
    ('browser', 99),                         # outside the dataset's code range
    ('operatingsystems', 0),                 # codes start at 1
    ('weekend', 'maybe'),                    # not a boolean
])
def test_invalid_values_are_rejected(client, payload, field, bad_value):
    response = client.post('/predict', json={**payload, field: bad_value})
    assert response.status_code == 422


@pytest.mark.parametrize('field', ['month', 'pagevalues', 'visitortype', 'weekend'])
def test_missing_fields_are_rejected(client, payload, field):
    del payload[field]
    assert client.post('/predict', json=payload).status_code == 422


def test_api_matches_training_pipeline(client, predict_module, clean_df):
    """The service must give the same probabilities as running the saved model on cleaned data."""
    fields = list(predict_module.Session.model_fields)
    sample = clean_df.iloc[::250]
    assert len(sample) > 30

    for _, row in sample.iterrows():
        body = {k: (row[k].item() if hasattr(row[k], 'item') else row[k]) for k in fields}
        for code in ['operatingsystems', 'browser', 'region', 'traffictype']:
            body[code] = int(body[code])          # the API takes integer codes
        body['weekend'] = bool(body['weekend'])

        api_prob = client.post('/predict', json=body).json()['purchase_probability']

        X = predict_module.DV.transform(train.to_dicts(row.to_frame().T))
        direct = float(predict_module.MODEL.predict_proba(X)[0, 1])
        assert api_prob == pytest.approx(direct, abs=1e-4)