# Build an ML Pipeline for Customer Salary Predictions

[**Project Description**](#project-description) | [**Install**](#install) | [**Data**](#data) | [**Train model**](#train-model) | [**Run sanity checks**](#run-sanity-checks) | [**Run tests**](#run-tests) | [**CI/CD**](#cicd) | [**Dockerize**](#dockerize)  | [**Request API**](#request-api) | [**Model Card**](#model-card) | [**Code Quality**](#code-quality)

## Project Description
Apply the skills acquired in this course to develop a classification model on publicly available Census Bureau data. You will create unit tests to monitor the model performance on various data slices. Then, you will deploy your model using the FastAPI package and create API tests. The slice validation and the API tests will be incorporated into a CI/CD framework using GitHub Actions.

Source code: [pthoa13/Customer-Salary-Prediction-Deploying-ML-Model-to-Cloud-MLOps](https://github.com/pthoa13/Customer-Salary-Prediction-Deploying-ML-Model-to-Cloud-MLOps)


```bash
projects
├── data
│   ├── census.csv
│   ├── census_clean.csv
├── images
│   ├── continuous_deployment.png
│   ├── continuous_integration.png
│   ├── live_get.png
│   ├── live_post.png
│   ├── local_post.png
│   └── settings_continuous_deployment.png
├── log
│   └── log
├── model
│   ├── label_encoder_dict.joblib
│   ├── std_scaler.bin
│   └── model.joblib
├── notebook
│   └── preprocess_data.ipynb
├── scripts
│   ├── clean_data.py
│   ├── feature_engineering.py
│   ├── model_function.py
│   └── train_model.py
├── tests
│   ├── test_api.py
│   └── test_model.py
├── utils
│   ├── constants.yaml
│   └── load_config.py
├── .env
├── .gitignore
├── docker-compose.yaml
├── Dockerfile
├── main.py
├── model_card.md
├── README.md
├── requirements.txt
├── sanitycheck.py
├── setup.py

8 directories, 31 files
```
| # | Feature               | Stack             |
|:-:|-----------------------|:-----------------:|
| 0 | Language              | Python            |
| 1 | Clean code principles | Autopep8, Pylint  |
| 2 | Testing               | Pytest            |
| 3 | Logging               | Logging           |
| 4 | Data versioning       | DVC               |
| 5 | Model versioning      | DVC               |
| 6 | Configuration         | YAML             |
| 7 | Development API       | FastAPI           |
| 8 | Dockerize             | Docker            |
| 9 | Cloud computing       | Render            |
| 10| CI/CD                 | Github Actions    |


## Install
```bash
pip install -r requirements.txt
```

## Data
### 1. Download data
```bash
data/census.csv
```
Link: https://archive.ics.uci.edu/ml/datasets/census+income

### 2. EDA
EDA in notebook: [![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F.svg?style=for-the-badge&logo=jupyter&logoColor=white)](Customer-Salary-Prediction-Deploying-ML-Model-to-Cloud-MLOps/notebook/preprocess_data.ipynb)


## Train model

```bash
python scripts/train.py
```

## Run sanity checks
```bash
python sanity_checks.py
```

## Run tests
```bash
pytest tests/
```

## Dockerize
```bash
docker-compose up -d --build
```

## Model Card
```
Customer-Salary-Prediction-Deploying-ML-Model-to-Cloud-MLOps/model_card.md
```

## Code Quality
Style Guide - Format your refactored code using PEP 8 – Style Guide. Running the command below can assist with formatting. To assist with meeting pep 8 guidelines, use autopep8 via the command line commands below:
```bash
autopep8 --in-place --aggressive --aggressive .
```

Style Checking and Error Spotting - Use Pylint for the code analysis looking for programming errors, and scope for further refactoring. You should check the pylint score using the command below.
```bash
pylint -rn -sn .
```
Docstring - All functions and files should have document strings that correctly identifies the inputs, outputs, and purpose of the function. All files have a document string that identifies the purpose of the file, the author, and the date the file was created.