"""Tests for the cleaning and training logic in train.py."""

import pickle
import shutil

import pandas as pd
import pytest

import train
from conftest import DATA_CSV


def raw_frame() -> pd.DataFrame:
    """Two rows shaped like the raw CSV: CamelCase headers, 'June', bool columns."""
    return pd.DataFrame({
        'Administrative': [0, 3],
        'PageValues': [0.0, 22.9],
        'Month': ['June', 'Nov'],
        'OperatingSystems': [1, 2],
        'Browser': [1, 2],
        'Region': [1, 3],
        'TrafficType': [1, 2],
        'VisitorType': ['Returning_Visitor', 'New_Visitor'],
        'Weekend': [False, True],
        'Revenue': [False, True],
    })


def test_clean_lowercases_column_names():
    cleaned = train.clean(raw_frame())
    assert all(c == c.lower() and ' ' not in c for c in cleaned.columns)
    assert 'pagevalues' in cleaned.columns


def test_clean_standardizes_string_values_and_fixes_june():
    cleaned = train.clean(raw_frame())
    assert list(cleaned['month']) == ['jun', 'nov']
    assert list(cleaned['visitortype']) == ['returning_visitor', 'new_visitor']


def test_clean_turns_codes_into_strings_and_bools_into_ints():
    cleaned = train.clean(raw_frame())
    for col in ['operatingsystems', 'browser', 'region', 'traffictype']:
        assert all(isinstance(v, str) for v in cleaned[col])
    assert list(cleaned['weekend']) == [0, 1]
    assert list(cleaned['revenue']) == [0, 1]


def test_full_dataset_after_cleaning(clean_df):
    assert len(clean_df) == 12330
    assert clean_df.isna().sum().sum() == 0
    assert set(clean_df['month'].unique()) == {
        'feb', 'mar', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'}
    assert clean_df['revenue'].mean() == pytest.approx(0.1547, abs=1e-3)


def test_features_exclude_the_target(clean_df):
    rows = train.to_dicts(clean_df.drop(columns=['revenue']))
    assert 'revenue' not in rows[0]
    assert set(rows[0]) == set(train.CATEGORICAL + train.NUMERIC)


def test_main_trains_and_saves_a_working_artifact(tmp_path, monkeypatch, capsys):
    """End to end: train.py reads the CSV, trains, prints a good test AUC and saves model.bin."""
    (tmp_path / 'data').mkdir()
    shutil.copy(DATA_CSV, tmp_path / 'data' / 'online_shoppers_intention.csv')
    monkeypatch.chdir(tmp_path)

    train.main()

    auc_line = next(line for line in capsys.readouterr().out.splitlines()
                    if line.startswith('test ROC AUC'))
    assert float(auc_line.split()[3]) > 0.92

    with open(tmp_path / 'model.bin', 'rb') as f:
        artifact = pickle.load(f)
    assert set(artifact) == {'dv', 'model', 'threshold'}
    assert artifact['threshold'] == pytest.approx(0.15)
    assert 'revenue' not in artifact['dv'].get_feature_names_out()