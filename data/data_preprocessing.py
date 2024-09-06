from typing import AnyStr

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import Strategy


class DataProcessing:
    """
    A clas for Data preprocessing, slicing and splitting
    """
    @classmethod
    def handle_missing_data(cls, df: "pd.DataFrame", strategy: Strategy = Strategy.Mean) -> "pd.DataFrame":
        """
        Fills missing data in the DataFrame with the specified strategy.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing missing data to be handled.
        strategy : Strategy, optional
            The strategy to use for filling missing data. Default is Strategy.Mean.
            Available strategies:
            - Strategy.Mean: Fill missing values with the mean of the column.
            - Strategy.Median: Fill missing values with the median of the column.
            - Strategy.Mode: Fill missing values with the mode of the column.
            - Strategy.Drop: Drop rows with missing values.

        Returns
        -------
        pd.DataFrame
            The DataFrame with missing data handled according to the specified strategy.

        Raises
        ------
        ValueError
            If an unsupported strategy is provided.
        """
        if strategy == Strategy.Mean.value:
            return df.fillna(df.mean())
        elif strategy == Strategy.Median.value:
            return df.fillna(df.median())
        elif strategy == Strategy.Mode.value:
            return df.fillna(df.mode().iloc[0])
        elif strategy == Strategy.Drop.value:
            return df.dropna()
        else:
            raise ValueError("Unsupported strategy. Use 'Mean' or 'Median'.")

    @classmethod
    def split_dataset(cls, df: "pd.DataFrame", target_column: AnyStr, test_size: float = 0.3):
        """
        Splits the dataset into training and testing sets.

        Parameters
        ----------
        df : pd.DataFrame
            The input dataframe containing the dataset.
        target_column : AnyStr
            The name of the target column in the dataframe.
        test_size : float, optional
            The proportion of the dataset to include in the test split (default is 0.3).

        Returns
        -------
        x_train : pd.DataFrame
            The training set features.
        x_test : pd.DataFrame
            The testing set features.
        y_train : pd.Series
            The training set target.
        y_test : pd.Series
            The testing set target.
        """
        x = df.drop(columns=[target_column])
        y = df[target_column]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
        return x_train, x_test, y_train, y_test

    @classmethod
    def slice_data(cls, df: "pd.DataFrame", condition) -> "pd.DataFrame":
        """
        Slices the DataFrame based on a given condition.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be sliced.
        condition : str
            The condition to query the DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame that meets the specified condition.
        """
        return df.query(condition)
