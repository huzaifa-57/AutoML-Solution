from typing import Any, Dict, AnyStr, Union

from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error


class Evaluation:
    """
    Class for generating evaluation metrics such as classification report, confusion matrix, and root mean square error.

    Examples
    --------
    >>> classification_report = Evaluation.generate_classification_report(y_test, y_pred)
    """
    @classmethod
    def generate_classification_report(cls, y_true, y_pred) -> Union[AnyStr, Dict]:
        """
        Generate a classification report.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Ground truth (correct) target values. array-like
        y_pred : array-like of shape (n_samples,)
            Estimated targets as returned by a classifier. array-like

        Returns
        -------
        AnyStr | dict
            Return string or dict of the precision, recall, F1 score for each class.
        """
        return classification_report(y_true, y_pred)

    @classmethod
    def generate_confusion_matrix(cls, y_true, y_pred):
        """
        Generate a confusion matrix for the given true and predicted labels.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True labels of the data. array-like
        y_pred : array-like of shape (n_samples,)
            Predicted labels of the data. array-like

        Returns
        -------
        ndarray of shape (n_classes, n_classes). A Confusion matrix.
        """
        return confusion_matrix(y_true, y_pred)

    @classmethod
    def calculate_root_mean_square_error(cls, y_true, y_pred) -> float:
        """
        Calculate Root Mean Squared error.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True labels of the data. array-like
        y_pred : array-like of shape (n_samples,)
            Predicted labels of the data. array-like


        Returns
        -------
        float
            Return calculated root mean squared error.
        """
        return mean_squared_error(y_true, y_pred, squared=False)
