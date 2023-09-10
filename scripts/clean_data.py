import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import logging
import pandas as pd
from utils.load_config import config


def load_data(filename):
    """
    Load the data from a CSV file.
    Inputs
    ------
        filename (str): The filename of the CSV file.
    Returns
    -------
        pandas.DataFrame: The data as a Pandas DataFrame.
    """

    try:
        df = pd.read_csv(filename)
        logging.info("SUCCESS: Data successfully imported")
        return df
    except Exception:
        logging.error("ERROR: Failed to load data")
        
def clean_data(filename, save=False):
    """
    Clean the data by removing spaces, removing duplicates, 
    replacing missing values, and reducing categories.
    Inputs
    ------
        filename (str): The filename of the CSV file.

    Returns
    -------
        pandas.DataFrame: The cleaned data as a Pandas DataFrame.
    """
    try: 
        # Load the data
        df = load_data(filename)
        
        # Remove spaces in columns name
        df = df.rename(columns = lambda x: x.strip())

        # Remove spaces in values
        df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)

        # Remove duplicates
        df = df.drop_duplicates()
        
        # Replacing Missing Data
        df['workclass'] = df['workclass'].replace("?", "Private")
        df['occupation'] = df['occupation'].replace("?", "Prof-speciality")
        df['native-country'] = df['native-country'].replace("?", "United-States")
        
        # Reducing Categories
        df['workclass'] = df['workclass'].replace(config["WORKCLASS_MAPPING"])
        df['education'] = df['education'].replace(config["EDUCATION_MAPPING"])
        df['marital-status'] = df['marital-status'].replace(config["MARTIAL_MAPPING"])
        df['relationship'] = df['relationship'].replace(config["RELATIONSHIP_MAPPING"])
        df['race'] = df['race'].replace(config["RACE_MAPPING"])
        
        # Save Clean Data
        if save:
            df.to_csv(config['CLEANED_DATA_PATH'], index=False)
        
        # Logging
        logging.info("SUCCESS: Save Clean Data")
        
        return df
    
    except Exception:
        logging.error("ERROR: Failed to clean data")

def preprocess_df(df):
    """
    Preprocess the DataFrame by replacing missing values and reducing the number of categories.

    Args:
        df (DataFrame): The DataFrame to be preprocessed.

    Returns:
        DataFrame: The preprocessed DataFrame.
    """
    # Fill Missing Values
    df['workclass'] = df['workclass'].replace("?", "Private")
    df['occupation'] = df['occupation'].replace("?", "Prof-speciality")
    df['native-country'] = df['native-country'].replace("?", "United-States")
    
    # Reducing Categories
    df['workclass'] = df['workclass'].replace(config["WORKCLASS_MAPPING"])
    df['education'] = df['education'].replace(config["EDUCATION_MAPPING"])
    df['marital-status'] = df['marital-status'].replace(config["MARTIAL_MAPPING"])
    df['relationship'] = df['relationship'].replace(config["RELATIONSHIP_MAPPING"])
    df['race'] = df['race'].replace(config["RACE_MAPPING"])
    
    return df

if __name__ == "__main__":
    df = clean_data(config['DATA_PATH'])