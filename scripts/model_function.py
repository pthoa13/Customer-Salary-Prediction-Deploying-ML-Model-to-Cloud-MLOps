import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import logging
from joblib import dump
from utils.load_config import config
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
            fbeta_score, 
            precision_score, 
            recall_score, 
            accuracy_score
)

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    try:
        model = GradientBoostingClassifier(**config["MODEL_PARAMS"])
        model.fit(X_train, y_train)
        logging.info("SUCCESS: trained model")
        return model
    except Exception:
        logging.info("ERROR: Failed to train model")


def compute_model_metrics(y, preds, data_slice=None, slicing=False ):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    data_slice : tuple (default: None)
        Tuple of slicing categories
    slicing : bool
        Whether computing the model or slicing
    Returns
    -------
    accuracy : float
    precision : float
    recall : float
    fbeta : float
    """
    tabs_string = "\t".join(["" for i in range(13)])
    # try:
    accuracy = accuracy_score(y, preds)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    if not slicing:
        logging.info(
            "SUCCESS: Model scoring computed.\n"
            "{tab}Accuracy: {accuracy}\n"
            "{tab}Precision: {precision}\n"
            "{tab}Recall: {recall}\n"
            "{tab}FBeta: {fbeta}".format(
                tab=tabs_string,
                accuracy=accuracy, 
                precision=precision, 
                recall=recall, 
                fbeta=fbeta
)
)
    else:
        logging.info(
            "SUCCESS: Model slicing {} -> {} scoring computed.\n"
            "{tab}Accuracy: {accuracy}\n"
            "{tab}Precision: {precision}\n"
            "{tab}Recall: {recall}\n"
            "{tab}FBeta: {fbeta}".format(
                data_slice[0], 
                data_slice[1], 
                tab=tabs_string,
                accuracy=accuracy, 
                precision=precision, 
                recall=recall, 
                fbeta=fbeta
            )
)
        
    return accuracy, precision, recall, fbeta
    # except Exception:
    #     logging.info("ERROR: Error when scoring model")


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model 
        Trained machine learning model.
    X: np.array
        Data used for prediction.
    Returns
    -------
    preds: np.array
        Predictions from the model.
    """
    try:
        preds = model.predict(X)
        logging.info("SUCCESS: Inference successfully")
        return preds
    except Exception:
        logging.info("ERROR: Error when inferencing model")

def save_model(model):
    """
    Save the model to a file.

    Inputs
    ------
        model: The model to save.
    Returns
    -------
        None.
    """
    try:
        dump(model, config["SAVED_MODEL_PATH"])
        logging.info("SUCCESS: Saved model")
    except Exception:
        logging.info("ERROR: Error when saving model")
    