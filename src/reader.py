import pandas as pd
import os
from copy import copy
from abc import ABC, abstractmethod
from typing import Optional, List, Callable, Iterator, Any


class Reader(ABC):

    def __init__(self, folder: str, file_filter: Optional[Callable] = None, sorted_key: Optional[Callable] = None):
        self._files = sorted(self._get_files_paths(folder, file_filter), key=sorted_key)
        self._iterator = self._create_iterator()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, item):
        other = copy(self)
        other._files = other._files[item]
        return other

    @staticmethod
    def _get_files_paths(folder: str, file_filter: Callable) -> List[str]:
        paths_list = []
        for root, folders_list, files_list in os.walk(os.path.abspath(folder)):
            for file in files_list:
                paths_list.append(os.path.join(root, file))
        if file_filter is not None:
            paths_list = list(filter(file_filter, paths_list))
        return paths_list

    @abstractmethod
    def _create_iterator(self) -> Iterator[Any]:
        raise NotImplemented


class CSVReader(Reader):

    def __init__(self, folder: str,
                 file_filter: Optional[Callable] = None,
                 sorted_key: Optional[Callable] = None,
                 chunk_size: int = 10000,
                 batch_size: int = 1):
        super().__init__(folder, file_filter, sorted_key)
        self._chunk_size = chunk_size
        self._batch_size = batch_size
        self._batch = []

    def _create_iterator(self) -> Iterator[List[dict]]:
        for file_path in self._files:
            for data in pd.read_csv(file_path, error_bad_lines=False, chunksize=self._chunk_size):
                for d in data.iterrows():
                    d = d[1].to_dict()
                    batch = self._compile_batch(d)
                    if batch is not None:
                        yield batch

    def _compile_batch(self, data: dict):
        if len(self._batch) != self._batch_size:
            self._batch.append(data)
        else:
            result, self._batch = self._batch, []
            return result


if __name__ == "__main__":
    FOLDER = r"C:\Projects\INNC-MLFR\data\raw"
    reader = CSVReader(
        folder=FOLDER,
        file_filter=lambda x: "log" in x.split("_")
    )

    for r in reader:
        print(r)
