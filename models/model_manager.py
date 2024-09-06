from typing import Union, Tuple
import json

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV

from utils import ModelType


class CustomModel:
    """
    A class for training and evaluating custom models using RandomForestClassifier and RandomForestRegressor.
    """
    @classmethod
    def train_model(cls, x_train, y_train, model_type: ModelType = ModelType.Classification, params: str = None) -> Union[RandomForestClassifier, RandomForestRegressor]:
        """
        Train a model using RandomForestClassifier or RandomForestRegressor.

        Parameters
        ----------
        x_train : array-like of shape (n_samples, n_features). array-like
            The training input samples.
        y_train : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in regression). array-like
        model_type : ModelType, optional
            The type of model to train, either Classification or Regression. Default is Classification.
        params : str, optional
            A JSON string of hyperparameters for GridSearchCV. If None, default parameters are used.

        Returns
        -------
        model : Union[RandomForestClassifier, RandomForestRegressor]
            The trained model.
        """
        model = CustomModel.__get_model_by_factory(model_type)
        if params is not None and len(params) != 0:
            params = json.loads(params)
            model = GridSearchCV(model, params, cv=5)
        model.fit(x_train, y_train)
        return model
    
    @classmethod
    def evaluate_model(cls, model, x_test, y_test, model_type: ModelType = ModelType.Classification) -> Tuple[float, str]:
        """
        Evaluates the performance of a given model on test data.

        Parameters
        ----------
        model : object
            The trained model to be evaluated.
        x_test : array-like
            The test features.
        y_test : array-like
            The true labels for the test data.
        model_type : ModelType, optional
            The type of model being evaluated, either Classification or Regression.
            Default is ModelType.Classification.

        Returns
        -------
        Tuple[float, str]
            A tuple containing the evaluation metric and its name. For classification models,
            it returns the accuracy score and "Accuracy Score". For regression models, it returns
            the mean squared error (inverted and rounded) and "Mean Squared Error".

        Raises
        ------
        ValueError
            If the 'model_type' is not supported (neither Classification nor Regression).
        """
        predictions = model.predict(x_test)
        if model_type == ModelType.Classification.value:
            return accuracy_score(y_test, predictions), "Accuracy Score"
        elif model_type == ModelType.Regression.value:
            mse = mean_squared_error(y_test, predictions, squared=False)
            accuracy = float(mse).__round__(5)
            return accuracy, "Mean Squared Error"
        else:
            raise ValueError("Unsupported model type. Use 'Classification' or 'Regression'.")

    @staticmethod
    def __get_model_by_factory(model_type: ModelType = ModelType.Classification) -> Union[RandomForestClassifier, RandomForestRegressor]:
        """
        Factory method to get a model instance based on the specified model type.

        Parameters
        ----------
        model_type : ModelType, optional
            The type of model to create. Defaults to ModelType.Classification.

        Returns
        -------
        Union[RandomForestClassifier, RandomForestRegressor]
            An instance of RandomForestClassifier if model_type is ModelType.Classification,
            or an instance of RandomForestRegressor if model_type is ModelType.Regression.

        Raises
        ------
        ValueError
            If the 'model_type' is not supported (i.e., not Classification or Regression).
        """
        match model_type: # Alternate of "Switch-Case". Only Supports python >= 3.10
            case ModelType.Classification.value:
                model = RandomForestClassifier(random_state=101)
                return model
            case ModelType.Regression.value:
                model = RandomForestRegressor(random_state=101)
                return model
            case _:
                raise ValueError("Unsupported model type. Use 'Classification' or 'Regression'.")
