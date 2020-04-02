import pandas as pd
import numpy as np
import os
import random
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Optional, List, Callable, Iterator, Union, Any
import matplotlib.image as mpimg


class Reader(ABC):

    def __init__(self, paths: Union[str, List[str]],
                 file_filter: Optional[Callable] = None):
        paths = [paths] if isinstance(paths, str) else paths
        self.__files = self.__get_files_paths(paths, file_filter)

    def __iter__(self):
        return self._create_iterator(self.__files)

    def __len__(self):
        return len(self.__files)

    def __getitem__(self, item):
        other = deepcopy(self)
        other.__files = other.__files[item]
        return other

    def sort(self, key: Optional[Callable] = None):
        self.__files = sorted(self.__files, key=key)

    def shuffle(self):
        random.shuffle(self.__files)

    def endless_iterator(self, concatenate_method: Optional[Callable] = None,
                         batch_size: int = -1,
                         shuffle_files: bool = False):

        assert batch_size > 0 or batch_size == -1, f"Batch size should be > 0 or == -1, received {batch_size}"

        concatenate_method = (lambda x: x) if concatenate_method is None else concatenate_method
        batch = []

        while True:
            files = self.__files if not shuffle_files else random.sample(self.__files, len(self.__files))
            for step in self._create_iterator(files):
                if batch_size == -1:
                    yield step
                elif len(batch) == batch_size:
                    result, batch = batch, [step]
                    yield concatenate_method(result)
                else:
                    batch.append(step)

    def __get_files_paths(self, paths: List[str], file_filter: Callable) -> List[str]:
        paths_list = []

        for path in paths:
            path = os.path.abspath(path)
            if os.path.isfile(path):
                paths_list.append(path)
            elif os.path.isdir(path):
                paths_list += self.__parse_folder(path)

        if file_filter is not None:
            paths_list = list(filter(file_filter, paths_list))

        return paths_list

    @staticmethod
    def __parse_folder(folder: str):
        files = []
        for root, _, files_list in os.walk(folder):
            for file in files_list:
                files.append(os.path.join(root, file))
        return files

    @abstractmethod
    def _create_iterator(self, files_paths: List[str]) -> Iterator:
        raise NotImplemented


class CSVReader(Reader):

    def __init__(self, paths: str,
                 file_filter: Optional[Callable] = None,
                 chunk_size: int = 10000):

        if callable(file_filter):
            complex_filter = lambda x: file_filter(x) and x.split(".")[-1] == "csv"
        else:
            complex_filter = lambda x: x.split(".")[-1] == "csv"

        super().__init__(paths, complex_filter)
        self._chunk_size = chunk_size

    def _create_iterator(self, files_paths: List[str]) -> Iterator[dict]:
        """
        Create processing file iterator
        Returns:
            Iterator[dict]
        """
        for path in files_paths:
            for data in pd.read_csv(path, error_bad_lines=False, chunksize=self._chunk_size):
                for d in data.iterrows():
                    yield d[1].to_dict()


class ImageReader(Reader):

    def __init__(self, paths: str,
                 file_filter: Optional[Callable] = None):

        if callable(file_filter):
            complex_filter = lambda x: file_filter(x) and x.split(".")[-1] in {"jpg", "jpeg", "png", "tiff"}
        else:
            complex_filter = lambda x: x.split(".")[-1] in {"jpg", "jpeg", "png", "tiff"}

        super().__init__(paths, complex_filter)

    def _create_iterator(self, files_paths: List[str]) -> Iterator[np.ndarray]:
        for path in files_paths:
            yield mpimg.imread(path)


if __name__ == "__main__":
    DATA_PATH = r"C:\Projects\Customers\ConocoFillips\data\interim"

    data = ImageReader(
        paths=DATA_PATH,
        file_filter=lambda x: x.split(".")[-1] == "tiff"
    )
    for i in data[:6]:
        print(i.shape)