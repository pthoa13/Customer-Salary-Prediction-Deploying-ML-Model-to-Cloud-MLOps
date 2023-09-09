import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import joblib
import pandas as pd
import numpy as np
from utils.load_config import config
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler

def process_data(
    dataframe,
    label="salary",
    training=True,
):
    """ Process the data used in the machine learning pipeline.
    
    Inputs
    ------
    dataframe : pd.DataFrame
        Dataframe containing the features and label. 
        Columns in `categorical_features`
        
    label : str
        Name of the label column in `X`. 
        If None, then an empty array will be returned for y (default=None)
        
    training : bool
        Indicator if training mode or inference/validation mode.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    """
    cat_cols = config["CAT_FEATURES"]
    if training is True:
        le_dict = defaultdict(LabelEncoder) 
        df_fit = dataframe.apply(lambda x: le_dict[x.name].fit_transform(x) 
                                if x.name in cat_cols else x)
        x = df_fit.drop(label, axis=1)
        y = df_fit[label]
        joblib.dump(le_dict, config["LE_PATH"])
        
        # Normalize the Data 
        scaler = StandardScaler()
        x = pd.DataFrame(scaler.fit_transform(x), columns = x.columns)
        joblib.dump(scaler, config["SCALER_PATH"], compress=True)
    else:
        # Load the le dictionary
        le_dict = joblib.load(config["LE_PATH"])
        
        #Load Scaler
        scaler = joblib.load(config["SCALER_PATH"])
        
        df_fit = dataframe.apply(lambda x: le_dict[x.name].transform(x) 
                                if x.name in cat_cols else x)
        if label is not None:
            x = df_fit.drop(label, axis=1)
            y = df_fit[label]
        else:
            x = df_fit
            y = np.array([])
        x = pd.DataFrame(scaler.transform(x), columns = x.columns)
    
    return x, y
                       
            
if __name__ == "__main__":
    x_temp, y_temp = process_data(pd.read_csv("../data/cleaned_census.csv"),
                                  label="salary",
                                  training=True)