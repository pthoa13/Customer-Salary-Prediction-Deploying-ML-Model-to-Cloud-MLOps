import joblib
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

from utils.load_config import config
from scripts.clean_data import preprocess_df
from scripts.feature_engineering import process_data


# FastAPI
app = FastAPI()

class Customer(BaseModel):
    age: int = 39
    workclass: Literal[
        'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
        'Local-gov', 'Self-emp-inc', 'Without-pay']
    fnlgt: int = 338409
    education: Literal[
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
        'Some-college',
        'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
        '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    education_num: int = 13
    marital_status: Literal[
        'Never-married', 'Married-civ-spouse', 'Divorced',
        'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
        'Widowed']
    occupation: Literal[
        'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
        'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving',
        'Farming-fishing', 'Machine-op-inspct', 'Tech-support',
        'Craft-repair', 'Protective-serv', 'Armed-Forces',
        'Priv-house-serv']
    relationship: Literal[
        'Not-in-family', 'Husband', 'Wife', 'Own-child',
        'Unmarried', 'Other-relative']
    race: Literal[
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
        'Other']
    sex: Literal['Male', 'Female']
    capital_gain: int = 1319
    capital_loss: int = 0
    hours_per_week: int = 40
    native_country: Literal[
        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
        'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
        'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
        'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
        'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
        'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
        'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
        'Holand-Netherlands']
    
@app.get("/")
async def hello():
    return {"message": "Welcome!"}

@app.post("/predict_customer_salary")
async def predict_customer_salary(customer_data: Customer):
    
    # Load Model
    model = joblib.load(config["SAVED_MODEL_PATH"])
    
    #Loa Label Encoder
    le_dict = joblib.load(config["LE_PATH"])
    
    inp = np.array([[
                    customer_data.age,
                    customer_data.workclass,
                    customer_data.fnlgt,
                    customer_data.education,
                    customer_data.education_num,
                    customer_data.marital_status,
                    customer_data.occupation,
                    customer_data.relationship,
                    customer_data.race,
                    customer_data.sex,
                    customer_data.capital_gain,
                    customer_data.capital_loss,
                    customer_data.hours_per_week,
                    customer_data.native_country
    ]])
    
    df_temp = pd.DataFrame(data=inp, 
                           columns=[
                                "age",
                                "workclass",
                                "fnlgt",
                                "education",
                                "education-num",
                                "marital-status",
                                "occupation",
                                "relationship",
                                "race",
                                "sex",
                                "capital-gain",
                                "capital-loss",
                                "hours-per-week",
                                "native-country",
                           ])
    df_temp = preprocess_df(df_temp)
    X, _ = process_data(df_temp, training=False, label=None)
    pred = model.predict(X)
    result_df = pd.DataFrame(data=pred, columns=["salary"])
    result_df = result_df.apply(lambda x: le_dict[x.name].inverse_transform(x))
    result = result_df["salary"].values[0]
    return {
        "prediction": result 
    }