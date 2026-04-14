from dataclasses import dataclass

@dataclass
class BoundingBoxDegree:
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float

@dataclass
class BoundingBoxMeter:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

@dataclass
class DataDivision:
    """A class to define the division of data into training, validation, and test sets.

    All floats must be between 0 and 1 and their sum must equal 1.
    
    It is possible to set a paramter to 0, meaning that no data will be assigned to that set. 

    If `no_division` is set to True, all data will be written to a single folder (`output_path`) instead of being divided into train, val, and test folders.
    """
    train: float = 0
    val: float = 0
    test: float = 0
    no_division: bool = False

    def __post_init__(self):
        total = sum(self.__dict__.values())
        if self.no_division:
            return
        if abs(total - 1.0) > 1e-6:
            raise ValueError("The sum of non bool train, val, and test proportions must equal 1.")