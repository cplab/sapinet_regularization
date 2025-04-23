""" Enhancements to base Sapicore Data class. """
from __future__ import annotations
from typing import Any

import os

import numpy as np
import pandas as pd

import torch

from sapicore.data import Data, Metadata, AxisDescriptor
from sapicore.utils.io import ensure_dir


class SNNData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self, indices: Any = None):
        """Load data and labels to memory from the .pt files saved by :meth:`~data.Data._standardize`.
        Overwrites `buffer` and `descriptors`. If specified, selects only the rows at `indices`.

        Parameters
        ----------
        indices: slice, optional
            Specific indices to load.

        """
        # the labels file has been created by _standardize upon first obtaining the data.
        labels = torch.load(os.path.join(self.root, "labels.pt"))
        if indices is not None:
            labels = labels[indices]

        # by default, assume that all labels describe the 0th axis (rows).
        self.metadata = Metadata(*[AxisDescriptor(name=col, labels=labels[col].to_list(), axis=0) for col in labels])

        # load the data into the buffer (synthetic sets are typically small and don't require lazy loading).
        self.buffer = torch.load(os.path.join(self.root, "data.pt"))

        if indices is not None:
            self.buffer = self.buffer[indices]

        return self

    def save(self):
        """Dump the contents of the `buffer` and `metadata` table to disk at `destination`.
        This implementation saves the tensors as .pt files.

        """
        torch.save(self.buffer, os.path.join(ensure_dir(self.root), "data.pt"))
        torch.save(self.metadata.to_dataframe(), os.path.join(ensure_dir(self.root), "labels.pt"))

    def concatenate(self, other: SNNData):
        """Concatenates another dataset to this one, covering buffer and metadata labels alike."""
        # append other dataset's buffer to this one by vertically stacking.
        self.buffer = torch.cat((self[:], other[:]))

        # append values of shared metadata keys if found.
        for key in other.metadata.to_dataframe():
            if key in self.metadata.to_dataframe():
                self.metadata[key].labels = np.concatenate((self.metadata[key][:], other.metadata[key][:]))

    def select(self, conditions: str | list[str], axis: int = 0) -> slice | list:
        """Selects sample indices by descriptor values based on a pandas logical statement applied to one or more
        :class:`AxisDescriptor` objects, aggregated into a dataframe using :meth:`aggregate_descriptors`.

        Importantly, these filtering operations can be performed prior to loading any data to memory, as they only
        depend on :class:`AxisDescriptor` objects (labels) attached to this dataset.

        Note
        ----
        Applying selection criteria with pd.eval syntax is simple:

        * "age > 21" would execute `pd.eval("self.table.age > 21", target=df)`.
        * "lobe.isin['F', 'P']" would execute `pd.eval("self.table.lobe.isin(['F', 'P']", target=df)`.

        Where `df` is the table attribute of the :class:`~AxisDescriptor` the condition is applied to.

        Warning
        -------
        This method is general and may eventually be moved to the base :class:`~data.Data` class.

        """
        # wrap a single string condition in a list if necessary.
        if isinstance(conditions, str):
            conditions = [conditions]

        # aggregate relevant axis descriptors into a pandas dataframe (table).
        df = self.metadata.to_dataframe(axis=axis)

        # evaluate the expressions and filter the dataframe.
        for expression in conditions:
            parsed = "& ".join([("df." + expr).replace(". ", ".") for expr in expression.split("&")])
            parsed = "| ".join([("df." + expr).replace(". ", ".") for expr in parsed.split("|")])

            parsed = parsed.replace("df.df.", "df.")
            parsed = parsed.replace("df.(", "(df.")

            df = df[pd.eval(parsed, target=df).to_list()]

        # return indices where expressions evaluated to true.
        return df.index.to_list()
