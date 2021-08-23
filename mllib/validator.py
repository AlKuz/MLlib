import os
import re
import csv
import math

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Iterable, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RowWarning(object):
    row: int
    warnings: Dict[str, List[str]]


class Condition(ABC):

    @abstractmethod
    def check(self, x: Any) -> (bool, List[str]):
        raise NotImplementedError


class NumericCondition(Condition):

    def __init__(self, dtype: callable, max_val=None, min_val=None, allowed_vals: Optional[set] = None):
        self._dtype = dtype
        self._max = max_val
        self._min = min_val
        self._allowed_vals = allowed_vals if allowed_vals is not None else set()

    def check(self, x: Any) -> (bool, List[str]):
        result = []
        try:
            val = self._dtype(x)
        except ValueError as e:
            return False, [e]
        if self._max is not None and val > self._max:
            result.append(f"Value {val} is higher then allowed {self._max}")
        if self._min is not None and val < self._min:
            result.append(f"Value {val} is lower then allowed {self._max}")
        if self._allowed_vals and val not in self._allowed_vals:
            result.append(f"Value {val} is not allowed")
        if math.isnan(val) or val is None:
            result.append(f"Value {val} is not a number")
        return len(result) == 0, result


class StringCondition(Condition):

    def __init__(self, frmt: str = "", allowed_vals: Optional[set] = None):
        self._format = frmt
        self._allowed_vals = allowed_vals if allowed_vals is not None else set()

    def check(self, x: Any) -> (bool, List[str]):
        result = []
        if not re.match(self._format, x):
            result.append(f"Value {x} is not matched to '{self._format}'")
        if self._allowed_vals and x not in self._allowed_vals:
            result.append(f"Value {x} is not allowed")
        return len(result) == 0, result


class TimeCondition(Condition):

    def __init__(self, time_format: str = "%Y-%m-%d %H:%M:%S.%f"):
        self._time_format = time_format

    def check(self, x: Any) -> (bool, List[str]):
        result = []
        try:
            datetime.strptime(x, self._time_format)
        except ValueError as e:
            result.append(e)
        return len(result) == 0, result


class Validator(object):

    def __init__(self, conditions: Optional[Dict[str, Condition]] = None):
        self._conditions = dict() if conditions is None else conditions

    def validate(self, data: dict) -> Dict[str, List[str]]:
        warnings = dict()
        for col, condition in self._conditions.items():
            valid, info = condition.check(data[col])
            if not valid:
                warnings[col] = info
        return warnings


class CSVHandler(object):

    def __init__(self, validator: Validator):
        self._validator = validator

    def validate(self, path: str) -> Iterable[RowWarning]:
        """
        Валидация csv файла

        Ards:
            path (str): Путь к файлу
            sep (str): Разделитель значений
        Returns:
            list: Лог несоответствий
        """
        with open(os.path.abspath(path), 'r') as file:
            csv_reader = csv.DictReader(file)
            for r, file_row in enumerate(csv_reader):
                row_warnings = RowWarning(r + 2, self._validator.validate(file_row))
                if len(row_warnings.warnings) > 0:
                    yield row_warnings
