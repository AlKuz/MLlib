import logging

import joblib
import numpy as np
import os
import pandas as pd
import pickle

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union, Callable, Any, Optional, Literal
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

from data_exploration import DataExploration, CSVExploration
from custom_transformers import (
    RemoveColumnsTransformer,
    ProcessColumnsTransformer,
    ProcessDataFrameTransformer,
    FilterTransformer,
    AggregateTransformer,
    ValuesToNanTransformer,
    ResetIndexTransformer,
    DropRowsTransformer
)


class Data(ABC):

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def copy(self) -> "Data":
        """
        Make a deep copy of itself

        Returns:
            TableData: Deep copy of the TableTransformer object
        """
        return deepcopy(self)

    @abstractmethod
    def save_data(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def explore(self) -> DataExploration:
        raise NotImplementedError


class TableData(Data):

    def __init__(self, source: Union[str, pd.DataFrame], sep: str = ","):
        """
        Create new data transformer

        Args:
            source (str): CSV file path
            source (pd.DataFrame): Pandas data frame
            sep (str): Separator for CSV file if source is file path
        """
        super(TableData, self).__init__()
        self._data = self._load_data(source, sep)
        self._pipeline = list()

    @staticmethod
    def _load_data(source: Union[str, pd.DataFrame], sep: str = ",") -> pd.DataFrame:
        if isinstance(source, str):
            return pd.read_csv(source, sep=sep)
        elif isinstance(source, pd.DataFrame):
            return source
        else:
            raise TypeError(f"Unknown source object. Expected Union[str, pd.DataFrame], received {type(source)}")

    def explore(self) -> CSVExploration:
        return CSVExploration(self._data)

    @property
    def data(self):
        return self._data

    @property
    def pipeline(self) -> Pipeline:
        return Pipeline(self._pipeline)

    def save_data(self, path: str):
        self._data.to_csv(path, index=False)
        self._logger.info(f"Data was saved in '{path}'")

    def save_pipeline(self, path, file_type: Literal["pickle", "joblib"] = "joblib"):
        path = os.path.abspath(path)
        _, ext = os.path.splitext(path)
        if file_type == "pickle":
            assert ext == ".pickle"
            with open(path, "wb") as file:
                pickle.dump(Pipeline(self._pipeline), file)
        elif file_type == "joblib":
            assert ext == ".joblib"
            joblib.dump(Pipeline(self._pipeline), path)
        self._logger.info(f"Pipeline was saved in '{path}'")

    def __len__(self) -> int:
        return len(self._data)

    def __add__(self, other: "TableData") -> "TableData":
        """
        Magic method to concatenate two TableTransformers

        Args:
            other (TableData): Other transformer

        Returns:
            TableData: The concatenated transformer, return a new object
        """
        new = self.copy()
        new._data = new._data.append(other._data)
        return new

    def __mul__(self, other: "TableData") -> "TableData":
        """
        Magic method for outer join two tables on the common columns

        Args:
            other (TableData): Other transformer

        Returns:
            TableData: New table created from via joining two old tables
        """
        new = self.copy()
        common_columns = set(new._data.columns).intersection(other._data.columns)
        new._data = new._data.join(other._data, on=common_columns, how="outer")
        return new

    def remove_columns(self, columns: list[str]) -> "TableData":
        """
        Remove columns and add RemoveColumnsTransformer to pipeline

        Args
            columns (list[str]): Dataframe column names to remove
        """
        other = self.copy()
        transformer = RemoveColumnsTransformer(self._logger, columns)
        other._data = transformer.transform(other._data)
        self._pipeline.append((f"remove_columns_{'_'.join(columns)}", transformer))
        return other

    def process_columns(self, transformers: dict[str, Union[Callable, tuple[str, Callable]]]) -> "TableData":
        """
        Process columns

        Args:
            transformers (Callable): Function which apply to the current column
            transformers (tuple[str, Callable]): The source column and the function which apply to the column.
                New column will be created.

        Returns:
            TableData: Deepcopy of self with modified or added columns
        """
        other = self.copy()
        transformer = ProcessColumnsTransformer(self._logger, transformers)
        other._data = transformer.transform(other._data)
        self._pipeline.append((f"remove_columns_{'_'.join(transformers.keys())}", transformer))
        return other

    def process_dataframe(self, transformers: Union[Callable, dict[str, Callable]]) -> "TableData":
        """
        Add new column which is created via any dataframe columns

        Args:
            transformers (Callable[[pd.DataFrame], pd.DataFrame]): Function to apply to dataframe,
                return modified dataframe
            transformers (dict[str, Callable[[pd.DataFrame], Iterable]]): Function to apply to dataframe to make
                a new column

        Returns:
            TableData: Deepcopy of self with modified or added columns
        """
        other = self.copy()
        transformer = ProcessDataFrameTransformer(self._logger, transformers)
        other._data = transformer.transform(other._data)
        name = f"function_transform_{transformer.__name__}" \
            if isinstance(transformer, Callable) else \
            f"function_transform_column_{'_'.join(transformers.keys())}"
        self._pipeline.append((name, transformer))
        return other

    def filter(self, conditions: dict[str, Callable[[Any], bool]]) -> "TableData":
        """
        Filter data via several columns conditions

        Args:
            conditions (Callable[[Any], bool]): Function to filter values in the columns

        Returns:
            TableData: Filtered data
        """
        other = deepcopy(self)
        transformer = FilterTransformer(self._logger, conditions)
        other._data = transformer.transform(other._data)
        self._pipeline.append((f"filtered_columns_{'_'.join(conditions.items())}", transformer))
        return other

    def aggregate(self,
                  group_by: Union[str, list[str]],
                  apply_to: dict[str, tuple[str, Callable]]) -> "TableData":
        other = self.copy()
        transformer = AggregateTransformer(self._logger, group_by, apply_to)
        other._data = transformer.transform(other._data)
        self._pipeline.append((f"aggregated_columns_{'_'.join([a[0] for a in apply_to.values()])}", transformer))
        return other

    def reset_index(self) -> "TableData":
        other = self.copy()
        transformer = ResetIndexTransformer(self._logger)
        other._data = transformer.transform(other._data)
        self._pipeline.append(("reset_index", transformer))
        return other

    def split(self,
              delimiter: float,
              stratify_on: Optional[list[str]] = None) -> ("TableData", "TableData"):
        assert 0.0 < delimiter < 1.0
        if stratify_on is not None:
            stratify_on = self._data[stratify_on]
        left, right = train_test_split(self._data, train_size=delimiter, stratify=stratify_on)
        return TableData(left), TableData(right)

    def process_nan(self,
                    strategy: Literal["drop", "constant", "mean", "median", "most_frequent", "knn"] = "drop",
                    value: Optional[Any] = None,
                    critical_nan_rate: float = 0.8,
                    add_indicator: bool = False,
                    missing_values: Union[int, float, str, type(np.nan), type(None)] = np.nan) -> "TableData":
        other = self.copy()

        if type(missing_values) in [int, float, str] and strategy == "drop":
            transformer = ValuesToNanTransformer(self._logger, missing_values)
            other._data = transformer.transform(other._data)
            self._pipeline.append(("change_to_nan", transformer))

        nan_rate = other._data.isnull().sum() / len(other._data)
        columns_to_drop = [key for key, val in nan_rate.to_dict().items() if val > critical_nan_rate]

        if columns_to_drop:
            transformer = RemoveColumnsTransformer(self._logger, columns_to_drop)
            other._data = transformer.transform(other._data)
            self._pipeline.append(("drop_nan_columns", transformer))

        if strategy == "drop":
            transformer = DropRowsTransformer(self._logger)
            other._data = transformer.transform(other._data)
            self._pipeline.append(("drop_nan_rows", transformer))
        elif strategy == "knn":
            imp = KNNImputer(missing_values=missing_values, add_indicator=add_indicator)
            other._data = imp.fit_transform(other._data)
            self._pipeline.append(("knn_imputer", imp))
            self._logger.info("Nan values were filled by KNNImputer")
        else:
            imp = SimpleImputer(missing_values, strategy, value, add_indicator=add_indicator)
            other._data = imp.fit_transform(other._data)
            self._pipeline.append(("simple_imputer", imp))
            self._logger.info("Nan values were filled by SimpleImputer")

        return other

    def one_hot(self, columns: Optional[Union[str, list[str]]] = None) -> "TableData":
        other = self.copy()
        one_hot = OneHotEncoder()

        if columns is None:
            other._data = one_hot.fit_transform(other._data)
        else:
            columns = columns if isinstance(columns, list) else [columns]
            other._data = pd.concat([
                other._data.drop(columns, axis=1),
                one_hot.fit_transform(other._data[columns])
            ], axis=1)

        return other

    def feature_generating(self,
                           degree: int = 2,
                           columns: Optional[Union[str, list[str]]] = None
                           ) -> "TableData":
        other = self.copy()
        generator = PolynomialFeatures(degree)

        if columns is None:
            other._data = generator.fit_transform(other._data)
        else:
            columns = columns if isinstance(columns, list) else [columns]
            other._data = pd.concat([
                other._data.drop(columns, axis=1),
                generator.fit_transform(other._data[columns])
            ], axis=1)

        return other

    def feature_selection(self, target_column: str, bun_features: int = 5) -> "TableData":
        raise NotImplementedError

    def reduce_dimension(self):
        raise NotImplementedError
