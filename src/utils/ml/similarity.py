""" Representational similarity analysis utilities. """
from numpy.typing import NDArray

import os
import numpy as np

import torch
from torch import Tensor

from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.math import map_words_to_integers
from src.utils.data import SNNData


class RSA:
    def __init__(
        self,
        data: Tensor,
        labels: Tensor | NDArray,
        primary_cmap: str = "viridis",
        secondary_cmap: str = "rocket",
        class_cmap: str = "Spectral",
        identifier: str = "",
    ):
        self.data = data
        self.labels = labels

        self.primary_cmap = primary_cmap
        self.secondary_cmap = secondary_cmap
        self.class_cmap = class_cmap

        self.identifier = identifier

    def pairwise_matrix(self, render: bool = False, save_path: str = "", **kwargs) -> Tensor:
        """Pairwise distances."""
        # sort rows by class if asked.
        indices = np.argsort(self.labels)

        # compute pairwise distances.
        matrix = cdist(self.data[indices], self.data[indices])

        if save_path or render:
            plt.clf()

            colormap = plt.get_cmap(self.class_cmap)
            colors = [colormap(i) for i in np.linspace(0, 1, len(np.unique(self.labels)))]

            sns.clustermap(
                matrix,
                annot=False,
                cmap=plt.get_cmap(self.secondary_cmap),
                row_colors=[colors[int(i) - 1] for i in map_words_to_integers(self.labels[indices])],
                xticklabels=False,
                yticklabels=False,
                rasterized=True,
                figsize=(25, 25),
                **kwargs,
            )

            plt.grid(False)
            plt.title(f"{self.identifier} pairwise distances", loc="left")

            if save_path:
                plt.savefig(save_path)

            if render:
                plt.show()

        return torch.as_tensor(matrix)

    def distance_matrix(self, render: bool = False, save_path: str = "", **kwargs) -> (Tensor, float):
        """RSA Euclidean distance matrix, rendering optional. Also returns variance within classes as % of total."""
        dim = len(np.unique(self.labels))
        matrix = torch.zeros((dim, dim), device="cpu")

        for i in range(dim):
            for j in range(i, dim):
                matrix[i, j] = np.mean(
                    cdist(
                        self.data[np.where(self.labels == i + 1)[0], :], self.data[np.where(self.labels == j + 1)[0], :]
                    )
                )

        variance_within = torch.sum(matrix.diag()) / torch.sum(matrix)

        if save_path or render:
            # mirror upper triangle to lower and plot resulting distance matrix.
            matrix[np.tril_indices(dim, -1)] = matrix.T[np.tril_indices(dim, -1)]

            plt.clf()
            sns.heatmap(matrix, cmap=self.secondary_cmap, rasterized=True, **kwargs)

            plt.title(f"{self.identifier} dissimilarity by class")

            if save_path:
                plt.savefig(save_path)

            if render:
                plt.show()

        return matrix, variance_within


def representational_analysis(
    pipeline, layers: list[str], response_data: dict[str, SNNData], key: str = "", deformation: list = None
):
    def pairwise(rsa_object, render=pipeline.render):
        # depict the pairwise distances and their deltas as information flows from layer to layer.
        return rsa_object.pairwise_matrix(
            render=render,
            save_path=os.path.join(
                pipeline.out_dir,
                "plots" + (os.sep + f"F{pipeline.fold}" if pipeline.fold else ""),
                f"pairwise_{rsa_object.identifier}.{pipeline.image_format}",
            ),
        )

    def deformation_map(m1, m2, title, file, render=pipeline.render):
        plt.clf()

        plt.figure(figsize=pipeline.fig_size)
        sns.heatmap(m2 - m1, cmap="coolwarm", center=0, rasterized=True)

        plt.title(title)
        plt.savefig(
            os.path.join(pipeline.out_dir, "plots" + (os.sep + f"F{pipeline.fold}" if pipeline.fold else ""), file)
        )

        if render:
            plt.show()

    rsa_results = {}
    pairwise_results = {}

    for layer in layers:
        rsa_results[layer] = RSA(
            response_data[layer][:].cpu(),
            response_data[layer].metadata[key][:],
            primary_cmap=pipeline.primary_cmap,
            secondary_cmap=pipeline.secondary_cmap,
            class_cmap=pipeline.class_cmap,
            identifier=layer,
        )
        pairwise_results[layer] = pairwise(rsa_results[layer])

    if deformation:
        for pair in deformation:
            deformation_map(
                pairwise_results[pair[0]],
                pairwise_results[pair[1]],
                title=f"{pair[0]} to {pair[1]} deformation map, sorted by class",
                file=f"deformation_{pair[0]}_{pair[1]}.svg",
            )
