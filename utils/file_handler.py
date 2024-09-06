from typing import AnyStr, Union, Optional
import os
import shutil


class FileHandler:
    """
    A class for handling file operations including saving uploaded files.

    Examples
    --------
    >>> FileHandler.save_uploaded_file(file)
    """
    @classmethod
    def save_uploaded_file(cls, file: Union[AnyStr, bytes] = None, save_dir: Optional[str] = 'data/saved_data'):
        """
        Save an uploaded file to the specified directory.

        Parameters
        ----------
        file : Union[AnyStr, bytes], optional
            The file to be saved. It can be a file path (str) or file content (bytes). Default is None.
        save_dir : Optional[str], optional
            The directory where the file will be saved. Default is 'data/saved_data'. Optional

        Returns
        -------
        str
            The path to the saved file.

        Raises
        ------
        ValueError
            If the file is None.
        Exception
            If there is an error in saving the file.
        """
        try:
            if file is None:
                raise ValueError(f"File uploaded is not a valid file")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            dst_filepath = os.path.join(save_dir, os.path.basename(file) if isinstance(file, str) else "user_data.csv")

            if isinstance(file, str):
                shutil.copy(src=file, dst=dst_filepath)
            else:
                with open(dst_filepath, 'wb') as f:
                    f.write(file)
            return dst_filepath
        except Exception as ex:
            raise Exception(f"Error in saving file. Error: {str(ex)}")
