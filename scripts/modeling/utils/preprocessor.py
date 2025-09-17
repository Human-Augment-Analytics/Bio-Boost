'''
This module implements a Preprocessor class, which can be used to preprocess the CSV data (as stored in a pandas DataFrame). Right now, it's
essentially a wrapper around scikit-learn's StandardScaler class, but more pre-processing steps may be added in the future.
'''

from sklearn.preprocessing import StandardScaler
import pandas as pd

class Preprocessor:
    '''
    A class which can be used to transform the CSV dataset, as stored in a pandas DataFrame.
    '''

    def __init__(self):
        '''
        Initializes an instance of the Preprocessor class (essentially wraps a scikit-learn StandardScaler right now).

        Input: None.
        '''

        self.scaler = StandardScaler()

    def preprocess(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        '''
        Uses all the initialized preprocessing steps and procedures to preprocess the passed pandas DataFrame.

        Input:
            df: a pandas DataFrame, likely storing one of our CSV datasets.
            fit: a Boolean flag indicating whether or not self.scaler should also be fit on the passed pandas DataFrame (defaults to False).

        Output:
            scaled_df: a pandas DataFrame containing data which has been scaled according to the standard normal.
        '''
        
        if fit:
            scaled_df = self.scaler.fit_transform(df)
        else:
            scaled_df = self.scaler.transform(df)

        return scaled_df