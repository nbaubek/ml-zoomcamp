"""Train the final purchase-intent model and save it together with its preprocessing.

Reproduces the final model from notebook.ipynb (sections 2, 9, 12, 14, 15) with the
hyperparameters selected there by stratified 5-fold CV:

  - model: XGBoost (winner of the 4-model comparison)
  - params: n_estimators=100, learning_rate=0.05, max_depth=4, min_child_weight=10
  - features: full feature set including pagevalues (kept because every model without
    it fell below the 0.85 AUC bar; see notebook section 12)
  - decision threshold: 0.15, chosen from business costs (FN:FP = 5:1) on
    out-of-fold predictions; see notebook section 14

Run from the project root:  uv run python train.py
Produces model.bin (pickled dict with vectorizer, model and threshold).
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

RANDOM_STATE = 42
THRESHOLD = 0.15

CATEGORICAL = ['month', 'operatingsystems', 'browser', 'region', 'traffictype', 'visitortype']
NUMERIC = ['administrative', 'administrative_duration', 'informational', 'informational_duration',
           'productrelated', 'productrelated_duration', 'bouncerates', 'exitrates',
           'pagevalues', 'specialday', 'weekend']

BEST_PARAMS = {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 4, 'min_child_weight': 10}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Identical standardization to notebook.ipynb section 2 and predict.py."""
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    categorical_string_cols = [
        c for c in df.columns
        if df[c].dtype == 'object' or pd.api.types.is_string_dtype(df[c])
    ]
    for c in categorical_string_cols:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    # fix the inconsistent month label
    df['month'] = df['month'].replace({'june': 'jun'})

    # numeric codes are categories, not quantities -> strings for one-hot encoding
    for c in ['operatingsystems', 'browser', 'region', 'traffictype']:
        df[c] = df[c].astype(str)

    # bools -> ints
    df['weekend'] = df['weekend'].astype(int)
    df['revenue'] = df['revenue'].astype(int)
    return df


def to_dicts(df: pd.DataFrame) -> list[dict]:
    return df[CATEGORICAL + NUMERIC].to_dict(orient='records')


def main() -> None:
    df = pd.read_csv('data/online_shoppers_intention.csv')
    df = clean(df)

    # stratified 80/20 split, same seed as the notebook
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df['revenue'])
    df_full_train = df_full_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_full_train['revenue'].values
    y_test = df_test['revenue'].values
    df_train = df_full_train.drop(columns=['revenue'])
    df_test = df_test.drop(columns=['revenue'])

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(to_dicts(df_train))
    X_test = dv.transform(to_dicts(df_test))

    model = XGBClassifier(**BEST_PARAMS, random_state=RANDOM_STATE, n_jobs=-1,
                          eval_metric='logloss')
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_probs)
    pred = test_probs >= THRESHOLD
    tp = int((pred & (y_test == 1)).sum())
    fp = int((pred & (y_test == 0)).sum())
    fn = int((~pred & (y_test == 1)).sum())

    print(f'model: xgboost {BEST_PARAMS}')
    print(f'test ROC AUC:              {auc:.4f}   (notebook CV estimate: 0.9350)')
    print(f'at threshold {THRESHOLD}:')
    print(f'  precision {tp / (tp + fp):.3f}   recall {tp / (tp + fn):.3f}')
    print(f'  tp={tp}, fp={fp}, fn={fn}, tn={int((~pred & (y_test == 0)).sum())}')

    with open('model.bin', 'wb') as f:
        pickle.dump({'dv': dv, 'model': model, 'threshold': THRESHOLD}, f)
    print('saved model.bin')


if __name__ == '__main__':
    main()
