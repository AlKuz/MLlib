from abc import ABC, abstractmethod
from multiprocessing import Pool
from typing import Optional, Iterator, Iterable, Any


class AbstractPipeline(ABC):

    @abstractmethod
    def __call__(self, data: Iterable[Any]) -> Iterable[Any]:
        """
        Iteration step of the pipeline

        Args:
            data (Iterable[Any]): Batch of any artifacts

        Returns:
            Iterable[Any]: Processed batch
        """
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[Iterable[Any]]:
        """
        Create iterator from all pipeline steps or iterate on itself if it is the first step
        """

    @abstractmethod
    def __gt__(self, other: "AbstractPipeline") -> "AbstractPipeline":
        """
        Set next step of the pipeline: first_step > second_step > third_step

        Args:
            other (AbstractPipeline): Next step of the pipeline

        Returns:
            AbstractPipeline: Last step of the pipeline
        """
        return NotImplemented

    @abstractmethod
    def get_path(self) -> str:
        """
        Full path of the pipeline chain
        """
        raise NotImplementedError


class BasePipeline(AbstractPipeline):

    def __init__(self, workers: int = 0, ordered: bool = True):
        self._workers = workers
        self._ordered = ordered
        self._previous: Optional[AbstractPipeline] = None
        self._map = map if self._workers == 0 else self._pool_map

    def _pool_map(self, func, iterable):
        if self._ordered:
            return Pool(self._workers).imap(func, iterable)
        else:
            return Pool(self._workers).imap_unordered(func, iterable)

    def __call__(self, data: Iterable[Any]) -> Iterable[Any]:
        return data

    def __iter__(self) -> Iterator[Iterable[Any]]:
        return self._map(self.__call__, self._previous.__iter__)

    def __gt__(self, other: "AbstractPipeline") -> "AbstractPipeline":
        other._previous = self
        return other

    def get_path(self) -> str:
        if self._previous is None:
            return self.__class__.__name__
        else:
            return f"{self._previous.get_path()} > {self.__class__.__name__}"
