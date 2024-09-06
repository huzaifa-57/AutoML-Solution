# AutoML Solution

## Project Overview

This project is an AutoML solution that allows users to upload a tabular dataset (CSV file), preprocess the data, train machine learning models, and evaluate the models' performance. The solution provides an intuitive user interface built with Gradio, making it easy for users to interact with the AutoML features.

### Key Features:
- Upload CSV datasets for training
- Handle missing data with various strategies (mean, median, mode, or drop rows)
- Data slicing based on user-specified conditions
- Model Type selection for classification or Regression tasks
- Hyperparameter tuning with user-defined grids (JSON)
- Model evaluation with relevant metrics
- Simple and intuitive Gradio-based UI

## Installation and Setup

### 1. Prerequisites
- Ensure you have **Python 3.11.9** installed on your machine. You can verify your Python version by running:
  
  ```bash
  python --version
  ```
if you have not Python 3.11.9 then you can setup a virtual environment.

### 2. Setting up a virtual environment
To setup a virtual environment you can use either `conda` or `virtualenv`
- For conda, download miniconda3 by clicking [here](https://docs.anaconda.com/miniconda/)
- Then install the miniconda3.
- After installation, create a new environment by just pasting the below command in Command Prompt:
  ```commandline
  conda create -n "automl_solution" python=3.11.9
  ```
- The activate the bewly created environment.
  ```commandline
  conda activate "automl_solution"
  ```
- Install the required packages by going to the project directory in command prompt:
  ```commandline
  cd <PATH_TO_PROJECT_WHERE_YOU_DOWBLOADED_THE_ENTIRE_PROJECT>
  ```
### 3. Installing required packages
- Install the required packages by running below command:
  ```commandline
  pip install -r requirements.txt
  ```
### 4. Running the Application
- After successful installation of packages run the command below to start our project
  ```commandline
  python manager.py
  ```
  Click the link appear on the console. This will redirect you to the Gradio-based app. Now you can train and evaluate your model.