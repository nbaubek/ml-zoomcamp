"""FastAPI service: score an online-shop session with the purchase-intent model.

The saved model ranks sessions by purchase probability. The service applies the
identical standardization used at training time, then returns the probability and
the decision at the business threshold (0.15, notebook section 14).

Run locally:      uv run uvicorn predict:app --host 0.0.0.0 --port 9696
In Docker:        docker run -it --rm -p 9696:9696 purchase-intent-predictor
"""

import os
import pickle
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

# overridable so the tests can load a small throwaway model; defaults to the shipped one
MODEL_PATH = os.environ.get('MODEL_PATH', 'model.bin')

with open(MODEL_PATH, 'rb') as f_in:
    _artifact = pickle.load(f_in)

DV = _artifact['dv']
MODEL = _artifact['model']
THRESHOLD = _artifact['threshold']

app = FastAPI(
    title='purchase-intent-predictor',
    description='Scores online-shop sessions by probability of ending in a purchase.',
    version='1.0.0',
)


class Session(BaseModel):
    """Aggregated statistics of a browsing session, as captured by the shop's analytics.

    Field names and value ranges follow the UCI Online Shoppers dataset; the categorical
    code columns (OS, browser, region, traffic type) are integers exactly as in the data.
    """

    administrative: int = Field(ge=0, description='number of administrative pages visited')
    administrative_duration: float = Field(ge=0, description='seconds spent on administrative pages')
    informational: int = Field(ge=0, description='number of informational pages visited')
    informational_duration: float = Field(ge=0, description='seconds spent on informational pages')
    productrelated: int = Field(ge=0, description='number of product-related pages visited')
    productrelated_duration: float = Field(ge=0, description='seconds spent on product-related pages')
    bouncerates: float = Field(ge=0, le=1, description='average bounce rate of the pages visited')
    exitrates: float = Field(ge=0, le=1, description='average exit rate of the pages visited')
    pagevalues: float = Field(ge=0, description='Google Analytics page value of the pages visited')
    specialday: float = Field(ge=0, le=1, description='closeness of the visit date to a special day')
    month: Literal['feb', 'mar', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    operatingsystems: int = Field(ge=1, le=8, description='operating system code, as in the dataset')
    browser: int = Field(ge=1, le=13, description='browser code, as in the dataset')
    region: int = Field(ge=1, le=9, description='region code, as in the dataset')
    traffictype: int = Field(ge=1, le=20, description='traffic type code, as in the dataset')
    visitortype: Literal['returning_visitor', 'new_visitor', 'other']
    weekend: bool = Field(description='whether the visit date fell on a weekend')


class Prediction(BaseModel):
    purchase_probability: float
    purchase: bool
    threshold: float


@app.get('/health')
def health() -> dict:
    return {'status': 'ok'}


@app.post('/predict')
def predict(session: Session) -> Prediction:
    row = session.model_dump()

    # identical standardization to training (notebook section 2 / train.py):
    # code columns are categories -> strings; weekend stays a 0/1 numeric
    for c in ['operatingsystems', 'browser', 'region', 'traffictype']:
        row[c] = str(row[c])
    row['weekend'] = int(row['weekend'])

    X = DV.transform([row])
    prob = float(MODEL.predict_proba(X)[0, 1])

    return Prediction(
        purchase_probability=round(prob, 4),
        purchase=bool(prob >= THRESHOLD),
        threshold=THRESHOLD,
    )