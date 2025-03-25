from typing import Callable

import os
import numpy as np

import torch
from torch.nn.functional import normalize

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.datasets import make_classification

from sapicore.data.external.drift import DriftDataset
from sapicore.data.sampling import BalancedSampler

from src.utils.data import SNNData
from src.utils.data.synthesis import PhysicalSynthesizer, Synthetic

from src.utils.visual import Explorer
from src.utils.visual.embedding import SklearnEmbedder


def load_drift(
    pipeline, filter_conditions: list[str], sampler: Callable | BalancedSampler, shots: int = 1, folds: int = 2
) -> DriftDataset:
    """1: Ethanol; 2: Ethylene; 3:Ammonia; 4: Acetaldehyde; 5: Acetone; 6: Toluene."""
    drift = DriftDataset(identifier="UCSD drift", root=os.path.join(pipeline.data_dir, "drift")).load()
    # reps = self.configuration.get("simulation", {}).get("hetdup")

    # take the first feature (`DR`) from the first 8 sensors (there are 16 features X 8 non-redundant sensors).
    total_features = drift[:].shape[1]
    drift_subset = drift.sample(lambda: torch.arange(0, total_features // 2, 8), axis=1)

    # since the full set has 13910 entries, sample 2 of each batch X analyte combination (train and test).
    # sampling can be stratified with `n` representing a fraction of the total dataset.
    # here, users can specify a logical expression to filter classes or batches.
    key = pipeline.configuration.get("sampling", {}).get("key")
    group_key = pipeline.configuration.get("sampling", {}).get("group_key")

    drift_subset = drift_subset.trim(drift.select(filter_conditions))
    drift_subset = drift_subset.sample(method=sampler, group_keys=[key, group_key], n=(shots * folds))

    # move data tensor to GPU if necessary.
    drift_subset.buffer = drift_subset.buffer.to(pipeline.configuration.get("device", "cpu"))

    return drift_subset


def synthesize_data(
    pipeline,
    method: str,
    params: dict,
    sampler: Callable | BalancedSampler,
    duplicate: int = 0,
    norm: bool = False,
    key: str = "Category",
    group_keys: str | list[str] = None,
    shots: int = 1,
    folds: int = 2,
) -> SNNData:
    if method == "sklearn":
        data = Synthetic(algorithm=make_classification).generate(key=key, **params)

    elif method == "physical":
        data = PhysicalSynthesizer(**params).generate(key=key, n=params["n_samples"])

    else:
        raise ValueError(f"Invalid data synthesis setting: {method}.")

    data = data.sample(method=sampler, group_keys=group_keys, n=shots * folds)

    # optional, duplicate initially generated datapoints and their respective labels.
    temp_data = data[:]
    temp_labels = data.metadata[key].labels

    for _ in range(duplicate):
        data.buffer = torch.vstack((data[:], temp_data))
        data.metadata[key].labels = np.hstack((data.metadata[key][:], temp_labels))

    # shift data to positive range by adding a constant.
    min_value = torch.min(data[:])
    if min_value < 0:
        data[:] = data[:] - min_value

    # optional, L1-normalize the data.
    if norm:
        data[:] = normalize(data[:], p=1.0)

    # ensure data buffer tensor is on the correct device.
    data.buffer = data.buffer.to(pipeline.configuration.get("device", "cpu"))

    return data


def explore_data(
    pipeline,
    data: SNNData,
    proj: str = "diamond",
    key: str = "Category",
    interactive: bool = True,
    path: str = "",
    point_size: int = 10,
):
    emb = PCA(n_components=3, random_state=pipeline.seed)
    cmap = plt.get_cmap(pipeline.class_cmap, pipeline.configuration.get("synthesis", {}).get("classes"))

    explorer = Explorer(data=data, embedder=SklearnEmbedder(data=data, embedder=emb), fit=True, cmap=cmap)
    explorer.vista3d(
        key=key,
        style="points",
        point_size=point_size,
        render_points_as_spheres=True,
        projection=proj,
        interactive=interactive,
        path=path,
    )
