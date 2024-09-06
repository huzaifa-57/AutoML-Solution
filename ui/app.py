from typing import Any, List, Union, AnyStr

import gradio as gr

from utils import FileHandler, ModelType, Strategy
from data import CustomDataset, DataProcessing
from models import CustomModel


class AutoMLApp:
    """
    A Class to launch Gradio App
    """
    @classmethod
    def automl_interface(
            cls,
            uploaded_file: Union[str, bytes],
            target_column: AnyStr,
            test_size: float,
            model_type: ModelType,
            missing_strategy: Strategy,
            condition: AnyStr,
            params: AnyStr) -> str:
        """
        This function takes inputs. Load the Dataset. Slice and split the dataset. And then train and evaluate the model.
        And Show the performance of the model to the user.

        Parameters
        ----------
        uploaded_file : str | bytes
            take file path as an input or in bytes having content of the file
        target_column : str
            Name of the target Column on which the user wants the model to generate the predictions
        test_size : float
            The test size of the testing data. Range between 0.1 to 0.5 as 0,5 is max
        model_type : ModelType
            The type of model on which the user want to train his data. An Enum having valuees "Classification" or
            "Regression".
        missing_strategy : Strategy
            The strategy to fill the missing data in columns. Either it would be "Mean", "Median", "Mode" or "Drop". A enum
            having the same values.
        condition : AnyStr
            The condition to slice the data
        params : AnyStr
            The hyper parameters to fine tune the model.

        Returns
        -------
        AnyStr
            Returns a string which then plotted on output field. OR raised an exception if any error occured
        """
        try:
            if uploaded_file is None:
                return f"File has not been uploaded."
            filepath = FileHandler.save_uploaded_file(file=uploaded_file)
            df = CustomDataset.load_dataset(file_path=filepath)

            # Apply data preprocessing
            df = DataProcessing.handle_missing_data(df, strategy=missing_strategy)

            # Apply data slicing if a condition is specified
            if condition:
                df = DataProcessing.slice_data(df, condition)
            # Split the dataset
            x_train, x_test, y_train, y_test = DataProcessing.split_dataset(df, target_column, test_size=float(test_size))

            # Train the model
            model = CustomModel.train_model(x_train, y_train, model_type=model_type, params=params)

            # Evaluate the model
            performance, performance_measure = CustomModel.evaluate_model(model, x_test, y_test, model_type=model_type)
            return f"Model Performance - \"{performance_measure}\": {performance}"
        except Exception as ex:
            raise Exception(f"An unhandled Exception occurred. Error {str(ex)}")

    @classmethod
    def launch_app(cls):
        """
        This method launches the Gradio App
        """
        def input_fields() -> List:
            """
            Input Parameters for the Gradio Interfacer which also serve as UI.

            Returns
            -------
            List
                Return a List of input Paramters
            """
            # Define the Gradio UI components
            # with gr.Blocks() as interface:
            uploaded_file = gr.File(label="Upload a CSV File", file_types=[".csv"], type="filepath")
            target_column = gr.Textbox(label="Target Column", placeholder="Your Target Column name e.g Purchased", lines=1)
            test_size = gr.Slider(label="Test Size (0.1 - 0.5)", minimum=0.1, maximum=0.5, value=0.2)
            model_type = gr.Radio(label="Model Type", type="index", choices=[ModelType.Classification.name, ModelType.Regression.name])
            missing_strategy = gr.Radio(label="Missing Data Strategy", type="index", choices=[Strategy.Mean.name, Strategy.Median.name,
                                                                                              Strategy.Mode.name, Strategy.Drop.name])
            condition = gr.Textbox(label="Condition for Slicing data (optional)", placeholder="e.g., 'column_name > 50'")
            params = gr.Textbox(label="Hyperparameter Grid JSON (optional)", placeholder="Place your Hyperparameter Grid JSON of Random Forest here.")
            return [uploaded_file, target_column, test_size, model_type, missing_strategy, condition, params]

        def output_fields() -> List:
            """
            Fields where to show the output

            Return
            ------
            List
                Return a list of output Fields
            """
            output = gr.Textbox(label="Model Performance", placeholder="Model performance on your data...")
            return [output]


        interface = gr.Interface(fn=AutoMLApp.automl_interface, allow_flagging="never", inputs=input_fields(), outputs=output_fields(),
                                 title="AutoML Solution", description=f"This is the solution for AutoML Problem where we "
                                                                      f"train the Random Forest Classification and Regression "
                                                                      f"models on our DataSet. The Dataset must be in numeric "
                                                                      f"values not in string or object.")

        interface.launch()
