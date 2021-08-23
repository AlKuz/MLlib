import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from typing import Optional, Union, Callable, Literal


class DataExploration(object):
    pass


class CSVExploration(DataExploration):

    def __init__(self, data: pd.DataFrame, figsize: tuple[int, int] = (12, 10)):
        self._data = data
        self._figsize = figsize

    def general_info(self, num_examples: int = 0) -> pd.DataFrame:
        """
        Explore data

        Returns:
            pd.DataFrame: Data frame with info of source data
        """
        explore = pd.DataFrame({
            "dtypes": self._data.dtypes,
            "unique_#": [self._data[c].nunique() for c in self._data.columns],
            "null_%": [n / len(self._data) * 100 for n in self._data.isnull().sum()],
            "mean": self._data.mean(),
            "std": self._data.std(),
            "min": self._data.min(),
            **{f"{int(q * 100)}%": self._data.quantile(q) for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
            "max": self._data.max()
        })
        if num_examples > 0:
            examples = self._data.sample(num_examples, random_state=42).T
            return pd.concat([explore, examples], axis=1)
        else:
            return explore

    def pairplot(self):
        plt.figure(figsize=self._figsize)
        sns.pairplot(self._data)
        plt.show()

    def heatmap(self,
                columns: Optional[list[str]] = None,
                method: Union[Literal['pearson', 'kendall', 'spearman'], Callable] = "pearson"):
        plt.figure(figsize=self._figsize)
        corr = self._data.corr(method) if columns is None else self._data[columns].corr(method)
        sns.heatmap(corr, annot=True)
        plt.show()
