from typing import AnyStr, Union
import os

import pandas as pd


class CustomDataset:
    """
    Loads the dataset from a CSV file.
    """
    @classmethod
    def load_dataset(cls, file_path: Union[AnyStr, "os.PathLike"]) -> "pd.DataFrame":
        """
        Loads the dataset from a CSV file.

        Parameters
        ----------
        file_path : Union[AnyStr, os.PathLike]
            The path to the CSV file to be loaded.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the data from the CSV file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist or is not a file.
        ValueError
            If the file is not a valid CSV file.
        """
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            raise FileNotFoundError(f"file {file_path} does not exist.")
        if not file_path.endswith('.csv'):
            raise ValueError(f"File {file_path} is not a valis  '.csv' file.")
        return pd.read_csv(file_path)