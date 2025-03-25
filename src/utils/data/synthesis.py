""" Data synthesis. """
from typing import Any, Callable
from importlib import import_module

import torch
from torch import Tensor

import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

from scipy.stats import rv_continuous
from scipy.stats.distributions import expon, uniform

from sapicore.data import Metadata, AxisDescriptor
from src.utils.data import SNNData


class Synthetic:
    """Wraps a given data synthesis method and returns an enhanced Sapicore dataset with labels."""

    def __init__(self, algorithm: Callable):
        self.algorithm = algorithm

    def generate(self, key: str, *args, **kwargs) -> SNNData:
        """Default data and label generation method. Applies the given algorithm (Callable) under
        the assumption that it returns samples and labels."""
        samples, labels = self.algorithm(*args, **kwargs)
        return SNNData(buffer=torch.as_tensor(samples), metadata=Metadata(AxisDescriptor(name=key, labels=labels)))


class RandomVariable:
    def __init__(self, distribution: rv_continuous = expon, **kwargs):
        self.distribution = distribution
        self.kws = kwargs

    def generate(self, size: int = 1, normalize: bool = False):
        values = self.distribution.rvs(size=size, **self.kws)
        return values / norm(values, 1) if normalize else values


class SourceMixture:
    """Generates 'n' samples of length 'sources' given a scipy distribution and a set of design matrices.

    Samples are mixtures of sources whose quantities are expressed in arbitrary units. When a SourceMixture
    object is called with an integer 'n', it generates n samples in accordance with the design matrices
    given at initialization. The resulting matrix is meant to be multiplied by a transposed affinity matrix
    to obtain sensor responses.

    Experimental designs are expressed in terms of a condition X source X concentration (loc) matrix,
    a condition X source X spread (scale) matrix, and a scipy distribution common to all sources.
    In scipy, 'loc' determines characteristics like mean, peak, or lower boundary, depending on the
    specific distribution, while 'scale' determines standard deviation or range.

    Meant to be wrapped by utils.data.Synthetic to generate an SNNData set.

    """

    def __init__(self, locations: Tensor, scales: Tensor, distribution: rv_continuous = expon, n_samples: int = 1):
        # scipy distribution to use throughout.
        self.distribution = distribution

        # experimental design matrices, condition X source.
        self.locations = locations
        self.scales = scales

        # easy reference to numeric descriptors.
        self.n_conditions = locations.shape[0]
        self.n_sources = locations.shape[1]
        self.n_samples = n_samples

        if self.locations.shape != self.scales.shape:
            raise ValueError("Inconsistent source mixture design matrices.")

    def __call__(self, n: int = 1):
        """Generates 'n' samples per condition label present in the self.design dictionary."""
        samples = torch.zeros((self.n_conditions * self.n_samples, self.n_sources))
        labels = []

        for c in range(self.n_conditions):
            labels.extend([c + 1] * self.n_samples)
            for s in range(self.n_sources):
                rvs = RandomVariable(self.distribution, loc=self.locations[c, s], scale=self.scales[c, s])
                samples[c * self.n_samples : (c + 1) * self.n_samples, s] = torch.as_tensor(
                    rvs.generate(size=self.n_samples)
                )

        return samples, labels


class PhysicalSynthesizer(Synthetic):
    def __init__(
        self,
        seed_vector: Tensor,
        depth: int,
        sparsity: float,
        sensors: int,
        signal: dict[str, Any],
        contaminants: dict[str, Any],
        noise: dict[str, Any],
        sigmoid: dict[str, Any],
        **kwargs,
    ):
        # affinity and concentration configuration.
        self.seed_vector = torch.as_tensor(seed_vector)
        self.depth = depth
        self.sparsity = sparsity
        self.sensors = sensors

        self.signal = signal if signal else {}
        self.contaminants = contaminants if contaminants else {}
        self.noise = noise if noise else {}
        self.sigmoid = sigmoid if sigmoid else {}

        # store computed affinity matrix.
        self.affinities = torch.zeros((self.sensors, len(self.seed_vector)))

        super().__init__(algorithm=self.make_labeled_samples)

    @staticmethod
    def sharpen(array: Tensor, exp: float = 1, c: float = 0):
        """Sharpens an N-dimensional array by exponentiation-normalization, with additive modulation."""
        z = np.clip(array + c, 0, None)
        l1 = norm(z**exp, 1)

        return (z**exp) / (l1 if l1 != 0 else 1)

    @staticmethod
    def hoyer_measure(vector: Tensor) -> float:
        """Computes the Hoyer sparsity measure for the given vector or matrix, ranging between 0 and 1.

        Parameters
        ----------
        vector: NDArray
            Affinity tensor to compute the Hoyer sparsity for.

        Returns
        -------
        HoyerSparsity: float
            Hoyer measure of the tensor.

        """
        sqrt_n = np.sqrt(vector.shape[vector.dim() - 1])
        return (1 / (sqrt_n - 1)) * (
            sqrt_n - np.mean(norm(vector, 1, axis=vector.ndim - 1) / norm(vector, 2, axis=vector.dim() - 1))
        )

    @staticmethod
    def add_noise(matrix: Tensor, distribution: rv_continuous, proportion: float = 1, **kwargs):
        """Adds noise to 'proportion' of the 'matrix' rows, drawn from scipy 'distribution' with **kwargs.
        The noised indices are randomly selected.

        """
        n_replace = int(proportion * matrix.shape[matrix.dim() - 1])
        inds = np.random.choice(list(range(matrix.shape[matrix.dim() - 1])), n_replace, replace=False)

        for row in range(len(matrix)):
            matrix[row, inds] += torch.as_tensor(distribution.rvs(size=n_replace, **kwargs))

        return matrix

    def tune_contrast(
        self, matrix: Tensor, target: float = 0.5, eps: float = 0.0001, c_delta: float = 0.00001
    ) -> Tensor:
        """Sharpens a matrix row-wise to the desired sparsity.

        Parameters
        ----------
        matrix: Tensor
            The matrix to sharpen.

        target: float
            Desired Hoyer sparsity level to arrive at, ranging between 0 and 1 (defaults to 0.5).

        eps: float
            Error tolerance (defaults to .001).

        c_delta: float
            Value by which to change the additive constant with each iteration.

        """
        target = np.clip(target, 0, 1)
        temp = matrix

        def status(m: Tensor):
            if self.hoyer_measure(m) < target - eps:
                return -1
            elif self.hoyer_measure(m) > target + eps:
                return 1
            else:
                return 0

        # initial exponent and constant to start the search from.
        exp = 2
        const = 0

        overshot = False
        while exp > 1 and not overshot:
            # record current status.
            init_status = status(temp)

            # increment exponent if flat, decrement if sharp.
            exp -= init_status * 0.1

            # sharpen the matrix and check if gone past target sparsity in this iteration.
            temp = self.sharpen(matrix, exp=exp)
            overshot = init_status != status(temp)
        matrix = temp

        # fine tune with additive component.
        overshot = False
        while not overshot and status(temp) != 0:
            # record current status and increment additive component.
            init_status = status(temp)

            # positive constants flatten, negatives sharpen.
            const += init_status * c_delta

            # sharpen the matrix and check if gone past target sparsity in this iteration.
            temp = self.sharpen(matrix, exp=1, c=const)
            overshot = init_status != status(temp)
        matrix = temp

        return matrix

    def contrast_plot(self, vector: Tensor, cmap: str = "coolwarm"):
        """Visualize contrast curves for this synthetic data generation scenario."""
        vector /= norm(vector, 1)
        h = self.hoyer_measure(vector)

        targets = [h + i for i in np.arange(-0.3, 0.3, 0.01)]
        colors = [plt.get_cmap(cmap)(i) for i in np.linspace(0, 1, len(targets))]

        print(f"Initial Hoyer: {h}")
        plt.plot(vector, "-o", color="black", linewidth=8)

        for k, t in enumerate(targets):
            optimized = self.tune_contrast(vector, target=t)
            plt.plot(optimized, "-o", color=colors[k], alpha=0.7)

        plt.show()

    def make_affinities(self, vector: Tensor, depth: int):
        """Generates varied sensor affinities from a seed vector while retaining relative distances.

        Given a base vector and a depth parameter, recursively swap elements starting at the deepest level,
        saving the intermediate results as the affinities of different potential sensors.

        The goal is to ensure that each source has one sensor that prefers it to every other source, while
        respecting the original distances induced by the base vector. Meaning, if molecules A and B are very
        similar to each other and different to C and D, sensors with high affinity for the former will have
        low affinity for the latter. Within the group of AB sensors, some will prefer A and others B, preserving
        the distance between them in the base vector.

        Note
        ----
        This function generates N sensors for N sources, where N is len(base).

        """

        def _swap(mat, pivot: int):
            if pivot > 0:
                mat[pivot:, :pivot], mat[pivot:, pivot:] = mat[pivot:, pivot:], mat[pivot:, :pivot].copy()

                # recursive calls for each quadrant.
                r = np.vstack(
                    (
                        np.hstack((_swap(mat[:pivot, :pivot], pivot // 2), _swap(mat[:pivot, pivot:], pivot // 2))),
                        np.hstack((_swap(mat[pivot:, :pivot], pivot // 2), _swap(mat[pivot:, pivot:], pivot // 2))),
                    )
                )
                return r
            else:
                return mat

        self.affinities = _swap(np.array([vector] * (2**depth)), pivot=len(vector) // 2)

        while self.affinities.shape[0] < self.sensors:
            self.affinities = np.vstack((self.affinities, self.affinities))

    @staticmethod
    def sigmoid_sum(concentrations, affinities, translation, scale):
        result = torch.zeros((concentrations.shape[0], affinities.shape[0]))
        for i, x in enumerate(concentrations):
            for j, sensor in enumerate(affinities):
                result[i, j] = torch.sum((1 + torch.e ** (-scale * (x * affinities[j, :] - translation))) ** -1)

        return result

    def make_labeled_samples(self, n: int) -> (Tensor, Tensor):
        # sharpen or flatten the seed affinity vector to match specified target.
        tuned = self.tune_contrast(self.seed_vector, target=self.sparsity)
        n_diag = len(self.seed_vector)

        # generate affinities.
        self.make_affinities(tuned, self.depth)

        # if signal distribution parameters were specified as constants.
        if all([not isinstance(x, list) for x in [self.signal["loc"], self.signal["scale"]]]):
            # generate balanced analyte mixture samples (concentration on diagonal, 0 otherwise).
            location_matrix = np.eye(n_diag) * self.signal["loc"] + (1 - np.eye(n_diag)) * self.contaminants["loc"]

            # identity matrix for pure samples; 1s for "noise" amounts centered at 0 for off-diagonal sources.
            scale_matrix = np.eye(n_diag) * self.signal["scale"] + (1 - np.eye(n_diag)) * self.contaminants["scale"]

        # if signal distribution parameters were specified as lists.
        else:
            # NOTE for now, contaminant concentrations and variances are assumed constant across analytes.
            location_matrix = np.eye(n_diag) + (1 - np.eye(n_diag)) * self.contaminants["loc"]
            scale_matrix = np.eye(n_diag) + (1 - np.eye(n_diag)) * self.contaminants["scale"]

            np.fill_diagonal(location_matrix, self.signal["loc"])
            np.fill_diagonal(scale_matrix, self.signal["scale"])

        # create samples and labels from the above source mixture parameters, then generate affinities.
        signal_distribution = getattr(import_module("scipy.stats"), self.signal["distribution"])
        noise_distribution = getattr(import_module("scipy.stats"), self.noise["distribution"])

        samples, labels = SourceMixture(location_matrix, scale_matrix, signal_distribution, n_samples=n)()

        # drop some sensors to replicate the weak secondary sources problem.
        self.affinities = self.affinities[: self.sensors, :]

        # samples passed through sensors.
        if not self.sigmoid or (self.sigmoid.get("loc") == 0 and self.sigmoid.get("scale") == 0):
            samples = torch.matmul(samples[:], torch.as_tensor(self.affinities.transpose()))
        else:
            samples = self.sigmoid_sum(
                samples[:], torch.as_tensor(self.affinities), self.sigmoid["loc"], self.sigmoid["scale"]
            )

        if self.noise.get("proportion", 0) > 0:
            samples = self.add_noise(
                matrix=samples,
                distribution=noise_distribution,
                proportion=self.noise["proportion"],
                loc=self.noise["loc"],
                scale=self.noise["scale"],
            )

        return samples, labels
