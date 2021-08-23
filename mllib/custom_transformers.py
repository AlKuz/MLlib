import numpy as np
import pandas as pd

from logging import Logger
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Callable, Union, Any


class RemoveColumnsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, logger: Logger, columns: list[str]):
        self._logger = logger
        self._columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, data: pd.DataFrame, y=None):
        columns_to_drop = set(self._columns).intersection(set(data.columns))
        self._logger.info(f"Columns to drop: {', '.join(columns_to_drop)}")
        missed_columns = set(self._columns) - columns_to_drop
        self._logger.warning(f"Missed columns: {', '.join(missed_columns)}")
        data = data.drop(columns_to_drop, axis=1)
        self._logger.info(f"New data columns: {', '.join(data.columns)}")
        return data


class ProcessColumnsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, logger: Logger, transformers: dict[str, Union[Callable, tuple[str, Callable]]]):
        self._logger = logger
        self._transformers = transformers

    def fit(self, X, y=None):
        return self

    def transform(self, data: pd.DataFrame, y=None):
        for target, transformer in self._transformers.items():
            if isinstance(transformer, Callable):
                source, func = target, transformer
            elif (
                    isinstance(transformer, tuple) and
                    isinstance(transformer[0], str) and
                    isinstance(transformer[1], Callable)
            ):
                source, func = transformer
            else:
                message = f"Invalid data schema. Expected Union[Callable, tuple[str, Callable]], received {type(transformer)}"
                self._logger.error(message)
                raise TypeError(message)
            data[target] = data[source].map(func)
        return data


class ProcessDataFrameTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, logger: Logger, transformers: Union[Callable, dict[str, Callable]]):
        self._logger = logger
        self._transformers = transformers

    def fit(self, X, y=None):
        return self

    def transform(self, data: pd.DataFrame, y=None):
        if isinstance(self._transformers, Callable):
            data = self._transformers(data)
            self._logger.info(f"Data was transformed by '{self._transformers.__name__}' function")
        else:
            for target, func in self._transformers.items():
                data[target] = func(data)
                self._logger.info(f"Data was transformed by '{self._transformers.__name__}' function, "
                                  f"added new column '{target}'")
        return data


class FilterTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, logger: Logger, conditions: dict[str, Callable[[Any], bool]]):
        self._logger = logger
        self._conditions = conditions

    def fit(self, X, y=None):
        return self

    def transform(self, data: pd.DataFrame, y=None):
        for col, cond in self._conditions.items():
            data = data.loc[data[col].map(cond), :]
        self._logger.info(f"Data was filtered by columns {', '.join(self._conditions.keys())}")
        return data


class AggregateTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,
                 logger: Logger,
                 group_by: Union[str, list[str]],
                 apply_to: dict[str, tuple[str, Callable]]):
        self._logger = logger
        self._group_by = group_by
        self._apply_to = apply_to

    def fit(self, X, y=None):
        return self

    def transform(self, data: pd.DataFrame, y=None):
        grouped = data.groupby(self._group_by)
        name = f"{self._group_by} column" if isinstance(self._group_by, str) else f"{', '.join(self._group_by)} columns"
        self._logger.info(f"Data was grouped by {name}")
        data = pd.DataFrame(data={
            agg_name: grouped[column].aggregate(func) for agg_name, (column, func) in self._apply_to.items()
        })
        self._logger.info(f"Columns was grouped in {', '.join([a[0] for a in self._apply_to.values()])} columns")
        return data


class ResetIndexTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, logger: Logger):
        self._logger = logger

    def fit(self, X, y=None):
        return self

    def transform(self, data: pd.DataFrame, y=None):
        data = data.reset_index()
        self._logger.info("The index has been reset")
        return data


class ValuesToNanTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, logger: Logger, missing_values: Union[int, float, str, type(np.nan), type(None)]):
        self._logger = logger
        self._missing_values = missing_values

    def fit(self, X, y=None):
        return self

    def transform(self, data: pd.DataFrame, y=None):
        data = data.replace(self._missing_values)
        self._logger.info(f"Values equal '{self._missing_values}' was changed to None")
        return data


class DropRowsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, logger: Logger):
        self._logger = logger

    def fit(self, X, y=None):
        return self

    def transform(self, data: pd.DataFrame, y=None):
        origin_len = len(data)
        data = data.dropna()
        self._logger.info(f"Total number of removed Nan rows is {len(data) - origin_len}")
        return data
