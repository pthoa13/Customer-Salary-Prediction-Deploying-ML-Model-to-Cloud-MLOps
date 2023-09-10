import sys
import joblib
import pandas as pd
import pandas.api.types as pdtypes
import pytest
try:
    from scripts.clean_data import clean_data
    from scripts.feature_engineering import process_data
    from scripts.model_function import inference
    from scripts.train_model import split_data
    from utils.load_config import config
except ModuleNotFoundError:
    sys.path.append('./')
    from scripts.clean_data import clean_data
    from scripts.feature_engineering import process_data
    from scripts.model_function import inference
    from scripts.train_model import split_data
    from utils.load_config import config


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(config["CLEANED_DATA_PATH"], skipinitialspace=True)


def test_column_presence_and_type(data):
    """Tests that cleaned csv file has expected columns and types.

    Args:
        data (pd.DataFrame): Dataset for testing
    """

    required_columns = {
        "age": pdtypes.is_int64_dtype,
        "workclass": pdtypes.is_object_dtype,
        "fnlgt": pdtypes.is_int64_dtype,
        "education": pdtypes.is_object_dtype,
        "education-num": pdtypes.is_int64_dtype,
        "marital-status": pdtypes.is_object_dtype,
        "occupation": pdtypes.is_object_dtype,
        "relationship": pdtypes.is_object_dtype,
        "race": pdtypes.is_object_dtype,
        "sex": pdtypes.is_object_dtype,
        "capital-gain": pdtypes.is_int64_dtype,
        "capital-loss": pdtypes.is_int64_dtype,
        "hours-per-week": pdtypes.is_int64_dtype,
        "native-country": pdtypes.is_object_dtype,
        "salary": pdtypes.is_object_dtype,
    }

    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    # Check that the columns are of the right dtype
    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(
            data[col_name]
        ), f"Column {col_name} failed test {format_verification_funct}"


def test_column_values(data):
    # Check that the columns are of the right dtype
    for col_name in data.columns.values:
        assert not data[col_name].isnull().any(
        ), f"Column {col_name} has null values"


def test_inference(data):
    """
    Assert that inference function returns correct
    amount of predictions with respect to the input
    """
    # Load model
    model = joblib.load(config["SAVED_MODEL_PATH"])
    
    # Feature Engineering
    X, y = process_data(data,
                        label="salary",
                        training=True)

    # Train Test Split
    _, X_test, _, y_test = split_data(X, y)
    
    preds = inference(model, X_test)

    assert len(preds) == len(y_test)

