import pandas as pd
import os
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Optional, List, Callable, Iterator, Any


class Reader(ABC):

    def __init__(self, folder: str,
                 file_filter: Optional[Callable] = None,
                 sorted_key: Optional[Callable] = None,
                 batch_size: int = -1):
        assert batch_size > 0 or batch_size == -1, f"Batch size should be > 0 or == -1, received {batch_size}"
        self.__files = sorted(self.__get_files_paths(folder, file_filter), key=sorted_key)
        self.__batch_size = batch_size
        self.__batch = []

    def __iter__(self):
        for step in self._create_iterator(self.__files):
            batch = self.__compile_batch(step)
            if batch is not None:
                yield batch

    def __len__(self):
        return len(self.__files)

    def __getitem__(self, item):
        other = deepcopy(self)
        other.__files = other.__files[item]
        return other

    @staticmethod
    def __get_files_paths(folder: str, file_filter: Callable) -> List[str]:
        paths_list = []
        for root, folders_list, files_list in os.walk(os.path.abspath(folder)):
            for file in files_list:
                paths_list.append(os.path.join(root, file))
        if file_filter is not None:
            paths_list = list(filter(file_filter, paths_list))
        return paths_list

    def __compile_batch(self, data: Any):
        if self.__batch_size == -1:
            return data
        elif len(self.__batch) == self.__batch_size:
            result, self.__batch = self.__batch, []
            return result
        else:
            self.__batch.append(data)

    @abstractmethod
    def _create_iterator(self, files_paths: List[str]) -> Iterator:
        raise NotImplemented


class TimeSeriesCSVReader(Reader):

    def __init__(self, folder: str,
                 file_filter: Optional[Callable] = None,
                 sorted_key: Optional[Callable] = None,
                 batch_size: int = -1,
                 chunk_size: int = 10000):
        super().__init__(folder, file_filter, sorted_key, batch_size)
        self._chunk_size = chunk_size

    def _create_iterator(self, files_paths: List[str]) -> Iterator[dict]:
        """
        Create processing file iterator
        Returns:
            Iterator[dict]
        """
        for file_path in files_paths:
            for data in pd.read_csv(file_path, error_bad_lines=False, chunksize=self._chunk_size):
                for d in data.iterrows():
                    yield d[1].to_dict()





if __name__ == "__main__":
    FOLDER = r"C:\Projects\Customers\ConocoFillips\data\raw\crosslines"
    reader = TimeSeriesCSVReader(
        folder=FOLDER,
        file_filter=lambda x: "log" in x.split("_")
    )

    for r in reader:
        print(r)
