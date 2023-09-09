import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import logging

from utils.load_config import config
from scripts.clean_data import clean_data
from scripts.feature_engineering import process_data
from scripts.model_function import (
            train_model, 
            compute_model_metrics,
            inference,
            save_model
)
from sklearn.model_selection import train_test_split


# Set the logging level
logging.basicConfig(
        level=logging.DEBUG, 
        filename=config["LOG_PATH"],
        filemode="w",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def split_data(X, y):
    """
    Splits the data into train and test sets.

    Inputs:
    -------
        data (iterable): The data to split.

    Returns:
    -------
        (train, test): A tuple of (train_data, test_data).
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=0.3, random_state=13
                                    )
        logging.info("SUCCESS: Data was splitted successfully")
        return X_train, X_test, y_train, y_test
    except Exception:
        logging.info("ERROR: Data was not splitted successfully")
        
def model_slicing(model, data):
    """
    Examines the results of a model on different slices of data.

    Inputs:
    -------
        model: The model to evaluate.
        data: The data to evaluate the model on.

    Returns:
    -------
        dict: A dictionary of the results of the model on each slice.
    """
    logging.info("SUCCESS: Model slicing computing.")
    # Load the list of categorical features
    cat_features = config["CAT_FEATURES"]
    
    for feature in cat_features:
        for cls in data[feature].unique():
            df_temp = data[data[feature] == cls]
            X_test_temp, y_test_temp = process_data(
                                            dataframe=df_temp, 
                                            label="salary",
                                            training=False
                                        )
            y_preds = model.predict(X_test_temp)
            _, _, _, _ = compute_model_metrics(
                y_test_temp, y_preds, data_slice=(feature, cls), slicing=True)
            

if __name__ == '__main__':
    # Load & Clean Dataframe 
    df = clean_data(config["DATA_PATH"])
    
    # Feature Engineering
    X, y = process_data(df,
                        label="salary",
                        training=True)
    
    # Train Test Split
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Training Model
    model = train_model(X_train, y_train)
    
    # Save trained model
    save_model(model)
    
    # Compute and Logging Metrics
    pred = inference(model, X_test)
    accuracy, precision, recall, fbeta = compute_model_metrics(y_test, pred)
    
    # Logging score Model Slicing
    model_slicing(model, df)