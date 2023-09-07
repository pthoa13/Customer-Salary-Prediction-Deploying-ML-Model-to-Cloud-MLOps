import pickle
import pandas as pd
import numpy as np
from load_config import config
from sklearn.preprocessing import LabelEncoder, StandardScaler

def process_data(
    dataframe,
    label=None,
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
    
    if label is not None:
        x = dataframe.drop(label, axis=1)
        y = dataframe[label]
    else:
        y = np.array([])
     
     
    le_dict = {}   
    if training is True:
        # Iterate over the columns of the dataframe
        for column in x.columns:
            # Initialize the Label Encoder
            le = LabelEncoder()
            
            # Fit the label encoder object to the column
            x[column] = le.fit_transform(x[column])
            
            # Add the label encoder object to the dictionary
            le_dict[column] = le
        
        # Transform the label object
        le = LabelEncoder()
        y = le.fit_transform(y)
        le_dict["label"] = le
        
        # Save the dictionary to a file
        with open(config["LE_PATH"], 'wb') as f:
            pickle.dump(le_dict, f)
    
    else:
        # Load the le dictionary
        with open(config["LE_PATH"], 'rb') as f:
            le_dict = pickle.load(f)
        
        for column in x.columns:
            # Load the column's label encoder object
            le = le_dict[column]
            
            # Fit and transform the column 
            x[column] = le.transform(x[column])
        
        # Catch the exception when y in None when inference mode
        try:
            y = le_dict["label"].transform(y)
        except Exception:
            pass
    # Normalize the Data 
    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(x), columns = x.columns)
    
    return x, y
                       
            
if __name__ == "__main__":
    x_temp, y_temp = process_data(pd.read_csv("../data/cleaned_census.csv"),
                                  label="salary",
                                  training=True)