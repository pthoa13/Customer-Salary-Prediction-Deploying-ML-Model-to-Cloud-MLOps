import logging
import pandas as pd
from load_config import config

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
        
def clean_data(filename):
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
        df.to_csv(config['CLEANED_DATA_PATH'], index=False)
        
        # Logging
        logging.info("SUCCESS: Save Clean Data")
        
        return df
    
    except Exception:
        logging.error("ERROR: Failed to clean data")

if __name__ == "__main__":
    df = clean_data(config['DATA_PATH'])