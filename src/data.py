import pandas as pd
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE


URL = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'


def load_data():
    return pd.read_csv(URL)
    
    
def split_data(df):
    """
    Split the dataframe into train and test sets.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    return train_test_split(
        X, 
        y, 
        stratify=y, 
        test_size = 0.3, 
        random_state=RANDOM_STATE 
    )
