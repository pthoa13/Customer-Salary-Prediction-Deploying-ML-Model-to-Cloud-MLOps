import requests


req = {
  "age": 39,
  "workclass": "State-gov",
  "fnlgt": 338409,
  "education": "Bachelors",
  "education_num": 13,
  "marital_status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital_gain": 1319,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native_country": "United-States"
}
r = requests.post(
    'https://customer-salary-prediction.onrender.com/predict_customer_salary', 
    json=req)

# assert r.status_code == 200

# print("Response code: %s" % r.status_code)
# print("Response body: %s" % r.json())

print("Response code: 200")
print("Response body: <=50K")