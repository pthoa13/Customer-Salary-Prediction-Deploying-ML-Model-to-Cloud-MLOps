import sys
import pytest
from fastapi.testclient import TestClient

try:
    from main import app
except ModuleNotFoundError:
    sys.path.append('./')
    from main import app


@pytest.fixture(scope="session")
def client():
    client = TestClient(app)
    return client


def test_get(client):
    """Test standard get"""
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == {"message": "Welcome!"}


def test_post_above(client):
    """Test for salary above 50K"""

    res = client.post("/predict_customer_salary", json={
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Other-relative",
        "race": "White",
        "sex": "Female",
        "capital_gain": 14084,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    })

    assert res.status_code == 200
    assert res.json() == {'prediction': '>50K'}


def test_post_below(client):
    """Test for salary below 50K"""
    res = client.post("/predict_customer_salary", json={
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "United-States"
    })

    assert res.status_code == 200
    assert res.json() == {'prediction': '<=50K'}


def test_get_invalid_url(client):
    """Test invalid url"""
    res = client.get("/invalid_url")
    assert res.status_code == 404
    assert res.json() == {'detail': 'Not Found'}