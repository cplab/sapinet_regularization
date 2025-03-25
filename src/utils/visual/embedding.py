""" Plot and explore structural properties of your data. """
from typing import Any

from sklearn.base import BaseEstimator
from snn.utils.data import SNNData

__all__ = ("Embedder", "SklearnEmbedder")


class Embedder:
    """Abstract interface for dimensionality reduction operations."""

    def __init__(self, data: SNNData, algorithm: Any, **kwargs):
        self.data = data
        self.algorithm = algorithm

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def transform(self, *args, **kwargs) -> SNNData:
        """Transform the data from original space to `dimension` by applying `algorithm`."""
        raise NotImplementedError


class SklearnEmbedder(Embedder):
    def __init__(self, data: SNNData, embedder: BaseEstimator):
        super().__init__(data=data, algorithm=embedder)

    def transform(self) -> SNNData:
        return SNNData(buffer=self.algorithm.fit_transform(self.data[:].cpu()), metadata=self.data.metadata)
