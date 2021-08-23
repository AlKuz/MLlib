import os

from abc import abstractmethod
from IPython.display import display, Markdown
from typing import Optional

from .data_processors import Data, TableData


class Description(object):

    def __init__(self, filepath: str):
        self._filepath = filepath

    def get(self) -> str:
        with open(self._filepath, "r") as file:
            return file.read()

    def print(self):
        print(self.get())

    def display(self):
        display(Markdown(self.get()))


class Dataset(object):

    def __init__(self, dataset_path: str):
        self._dataset_path = dataset_path

    @property
    def name(self) -> str:
        return os.path.split(self._dataset_path)[1]

    @property
    def description(self) -> Description:
        return Description(os.path.join(self._dataset_path, "description.md"))

    @abstractmethod
    def load(self) -> Data:
        raise NotImplementedError


class CSVDataset(Dataset):

    def load(self) -> TableData:
        return TableData(os.path.join(self._dataset_path, "raw.csv"))


class DatasetSelector(object):

    def __init__(self, dataset_path: str):
        self._dataset_path = dataset_path

    def select(self) -> Dataset:
        return CSVDataset(self._dataset_path)


class DatasetLoader(object):

    def __init__(self, datasets: str = "../data"):
        self._datasets = os.path.abspath(datasets)

    @property
    def names(self) -> list[str]:
        return os.listdir(self._datasets)

    def __getitem__(self, dataset_name: str) -> Optional[Dataset]:
        path = os.path.join(self._datasets, dataset_name)
        return None if not os.path.exists(path) else DatasetSelector(path).select()
