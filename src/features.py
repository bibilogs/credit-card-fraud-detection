import numpy as np


def create_features(df):
    """
    Applies a logarithmic transformation to the 'Amount' column using
    'numpy.log1p', which helps reduce the skewness of transaction amounts.
    
    Returns:
        A copy of the input DataFrame with an additional 'Amount_log' feature.
    """

    df = df.copy()
    df['Amount_log'] = np.log1p(df['Amount'])
    
    return df