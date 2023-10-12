# For preprocessing
import pandas as pd
import numpy as np
# local imports
from src.config import logger

class ProcessTrainTest():
    """
    ProcessTrainTest is a class for preprocessing training and testing DataFrames.

    Parameters:
        train_df (pd.DataFrame): DataFrame containing training data.
        test_df (pd.DataFrame): DataFrame containing testing data.
    """
    def __init__(self, train_df, test_df):
        """
        Initialize a ProcessTrainTest instance.

        Parameters:
            train_df (pd.DataFrame): DataFrame containing training data.
            test_df (pd.DataFrame): DataFrame containing testing data.
        """
        self.train_df = train_df
        self.test_df = test_df

    def reset_index(self):
        """
        Reset the index and rename a specific column in the training and testing DataFrames.

        Returns:
            pd.DataFrame, pd.DataFrame: Updated training and testing DataFrames with the index reset and column renamed.
        """
        self.train_df = self.train_df.rename(columns = {"Unnamed: 0":"Id"})
        self.test_df = self.test_df.rename(columns = {"Unnamed: 0":"Id"})
        

        return self.train_df, self.test_df 

   