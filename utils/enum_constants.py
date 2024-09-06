from enum import Enum


class ModelType(Enum):
    """
    An enumeration to represent different types of models types
    """
    Classification = 0
    Regression = 1


class Strategy(Enum):
    """
    An Enumeration class that represents the different missing data filling strategy.
    """
    Mean = 0
    Median = 1
    Mode = 2
    Drop = 3