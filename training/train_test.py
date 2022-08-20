import numpy as np
import pandas as pd

# functions to test are imported from train.py
from train import split_data, train_model, get_model_metrics

"""A set of simple unit tests for protecting against regressions in train.py"""


def test_split_data():
    test_data = {
        'id': [0, 1, 2, 3, 4],
        'target': [0, 0, 1, 0, 1],
        'col1': [1, 2, 3, 4, 5],
        'col2': [2, 1, 1, 2, 1]
        }

    data_df = pd.DataFrame(data=test_data)
    data = split_data(data_df)

    # verify that columns were removed correctly
    assert "target" not in data[0].data.columns
    assert "id" not in data[0].data.columns
    assert "col1" in data[0].data.columns

    # verify that data was split as desired
    assert data[0].data.shape == (4, 2)
    assert data[1].data.shape == (1, 2)


def test_train_model():
    X_train = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    y_train = np.array([10, 9, 8, 8, 6, 5])
    data = {"train": {"X": X_train, "y": y_train}}

    reg_model = train_model(data, {"alpha": 1.2})

    preds = reg_model.predict([[1], [2]])
    np.testing.assert_almost_equal(preds, [9.93939393939394, 9.03030303030303])


def test_get_model_metrics():

    class MockModel:

        @staticmethod
        def predict(data):
            return ([8.12121212, 7.21212121])

    X_test = np.array([3, 4]).reshape(-1, 1)
    y_test = np.array([8, 7])
    data = {"test": {"X": X_test, "y": y_test}}

    metrics = get_model_metrics(MockModel(), data)

    assert 'mse' in metrics
    mse = metrics['mse']
    np.testing.assert_almost_equal(mse, 0.029843893480257067)