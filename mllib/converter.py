import pandas as pd
import os
from abc import ABC, abstractmethod
from datetime import timedelta, datetime
from typing import Dict, Iterable, Union, List, Iterator, Any, Optional, Callable

from multiprocessing import Pool


class Converter(ABC):

    def __init__(self, workers: int = 0, ordered: bool = True):
        """
        Class initialisation

        Args:
            workers (int): Number of workers
                if < 0: use all cpu cores
                if == 0: don't use multiprocessing
                if > 0: use specified number of workers
        """
        self._workers = workers if workers >= 0 else os.cpu_count()
        self._ordered = ordered
        self._map = map if self._workers == 0 else self._pool_map

    def _pool_map(self, func, iterable):
        """It does not work if call outside the method"""
        if self._ordered:
            return Pool(self._workers).imap(func, iterable)
        else:
            return Pool(self._workers).imap_unordered(func, iterable)

    @abstractmethod
    def __call__(self, data: Iterable[Any]) -> Iterable[Any]:
        """
        Iteration step of the converter

        Args:
            data (Iterable[Any]): Batch of any data

        Returns:
            Iterable[Any]: Processed batch
        """
        raise NotImplementedError

    def create_iterator(self, data: Iterable[Iterable[Any]]) -> Iterator[Iterable[Any]]:
        """
        Create converter iterator

        Args:
            data (Iterable[Iterable[Any]]): Iterable of batches

        Returns:
            Iterator[Iterable[Any]]: Iterator of batches
        """
        return self._map(self.__call__, data)


class ApplyConverter(Converter):

    def __init__(self, func: Callable,
                 workers: int = 0, ordered: bool = True):
        super().__init__(workers, ordered)
        self._func = func

    def __call__(self, data: Iterable[Any]) -> Iterable[Any]:
        return self._func(data)


class DataConverter(Converter):

    def __init__(self, keep_columns: Optional[List[str]] = None,
                 transform_columns: Optional[Dict[str, Callable]] = None,
                 add_columns: Optional[Dict[str, Callable]] = None,
                 column_types: Optional[dict] = None,
                 workers=0,
                 ordered: bool = True):
        """
        Initialization of data converter

        Args:
            keep_columns (Optional[List[str]])
        :param keep_columns:
        :param transform_columns:
        :param add_columns:
        :param column_types:
        :param workers:
        """

        super().__init__(workers, ordered)
        self._transform_columns = transform_columns if transform_columns is not None else dict()
        self._add_columns = add_columns if add_columns is not None else dict()
        self._keep_columns = set(keep_columns + list(self._transform_columns.keys()) + list(self._add_columns.keys()))\
            if keep_columns is not None else None
        self._column_types = column_types if column_types is not None else dict()

    def __call__(self, data_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, data in enumerate(data_batch):
            data = self._create_new_columns(data)
            data = self._transform(data)
            data = self._filter_keys(data)
            data = self._change_types(data)
            data_batch[i] = data
        return data_batch

    def _filter_keys(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self._keep_columns is not None:
            return {key: val for key, val in data.items() if key in self._keep_columns}
        else:
            return data

    def _transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key, func in self._transform_columns.items():
            data[key] = func(data[key])
        return data

    def _create_new_columns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key, func in self._add_columns.items():
            data[key] = func(data)
        return data

    def _change_types(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key, dtype in self._column_types.items():
            data[key] = dtype(data[key])
        return data


class DelayConverter(Converter):

    def __init__(self, delay: timedelta,
                 key_column: Union[str, List[str]],
                 timestamp_column: str,
                 timestamp_format: str = "%Y-%m-%d %H:%M:%S.%f"):
        """
        Args:
            delay (timedelta): Временной интервал внутри которого будут храниться данные по транзакциям в виде датафрейма
            key_column (Union[str, List[str]]): Названия столбцов по которым идёт группировка. Если str, то группировка
                по одному столбцу. Если List[str], то создаётся вложенный словарь с ключами из уникальных значений
                указанных в списке имён столбцов
            timestamp_column (str): Название столбца с отметками по времени
        """

        super().__init__()
        self._delay = delay
        self._key_column = [key_column] if isinstance(key_column, str) else key_column
        self._timestamp_column = timestamp_column
        self._timestamp_format = timestamp_format
        self._dd: Dict[str, List[dict]] = dict()

    def __call__(self, data_batch: List[Dict[str, Any]]) -> List[pd.DataFrame]:
        result_batch = []
        for data in data_batch:
            key = "_".join([str(data[k]) for k in self._key_column])
            timestamp = self._get_timestamp(data)

            try:
                self._dd[key].append(data)
                self._dd[key] = self._filter_time(self._dd[key], timestamp)
            except KeyError:
                self._dd[key] = [data]

            result_batch.append(pd.DataFrame(data=self._list2dict(self._dd[key])))

        return result_batch

    def _get_timestamp(self, data: dict):
        if not isinstance(data[self._timestamp_column], datetime):
            data[self._timestamp_column] = datetime.strptime(data[self._timestamp_column], self._timestamp_format)
        return data[self._timestamp_column]

    def _filter_time(self, data: List[dict], timestamp) -> List[dict]:
        return list(filter(lambda x: timestamp - x[self._timestamp_column] <= self._delay, data))

    @staticmethod
    def _list2dict(data: List[dict]) -> Dict[str, list]:
        new_data = {key: [val] for key, val in data[0].items()}
        for d in data[1:]:
            for key, val in new_data.items():
                new_data[key].append(d[key])
        return new_data


class AggregationConverter(Converter):

    def __init__(self, keep_columns,
                 timestamp_column,
                 aggregations: Dict[str, List[str]],
                 time_delays: List[timedelta],
                 workers: int = 0,
                 ordered: bool = True):

        super().__init__(workers, ordered)
        self._aggregations = aggregations
        self._keep_columns = keep_columns
        self._timestamp_column = timestamp_column
        self._time_delays = time_delays

    def __call__(self, data_batch: List[pd.DataFrame]) -> pd.DataFrame:
        result_batch = []
        for i, data in enumerate(data_batch):
            time_delayed_data = [self._create_time_delayed_data(data, time_delay) for time_delay in self._time_delays]
            time_delayed_data = pd.concat(time_delayed_data)
            time_delayed_data["batch_id"] = [i] * len(time_delayed_data)
            result_batch.append(time_delayed_data)
        result_batch = pd.concat(result_batch)
        result_batch = self._transform_time_delays(result_batch)
        return result_batch

    def _create_time_delayed_data(self, df: pd.DataFrame, time_delay: timedelta):
        start_time = pd.to_datetime(df[self._timestamp_column].values[-1]) - time_delay
        time_delayed_data = df.loc[df[self._timestamp_column] >= start_time]
        time_delayed_data["time_delays"] = [str(time_delay)] * len(time_delayed_data)
        return time_delayed_data

    def _transform_time_delays(self, data: pd.DataFrame):
        data = data.groupby(["batch_id", "time_delays"] + self._keep_columns).agg(self._aggregations)
        data = data.unstack("time_delays").reset_index(self._keep_columns)
        data = self._flatten_column_names(data)
        return data

    def _flatten_column_names(self, df: pd.DataFrame):
        new_columns_names = []
        for col, agg, time_delay in list(df.columns):
            if col in self._aggregations.keys():
                new_columns_names.append(f"{col}_{agg}_{time_delay}")
            else:
                new_columns_names.append(col)

        df.columns = new_columns_names
        return df
