""" Classification utilities. """
import csv
import os
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

import torch

from torch import Tensor
from torch.nn.functional import normalize

from scipy import stats

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, LeaveOneOut

from sapicore.data.sampling import CV
from sapicore.utils.io import ensure_dir

from src.utils.data import SNNData
from src.utils.ml.splitting import LimitedStratifiedKFold


class SVM:
    def __init__(self, identifiers: dict, data: Tensor, labels: Tensor | NDArray, cv: CV = None):
        self.identifiers = identifiers

        self.data = data
        self.labels = labels

        self.cv = LeaveOneOut() if cv is None else cv

        # fold tracker.
        self.fold = 0

    def score_svm(
        self,
        kernel: str = "rbf",
        c: float = 10**5,
        gamma="scale",
        seed=96,
        ci_alpha=0.95,
        flip=False,
        scale=False,
        norm=False,
        file=None,
    ):
        """Shorthand for fitting, testing, and scoring with a single SVM.

        Returns mean accuracy across folds, a confidence interval, and a confusion matrix.
        If `file` is not None, saves results to CSV.

        """
        svm = SVC(kernel=kernel, C=c, gamma=gamma, random_state=seed, probability=True, class_weight="balanced")

        scores = np.zeros(self.data.shape[0])
        mcc = np.zeros(self.data.shape[0])
        # extended = []

        classes = len(np.unique(self.labels))
        confusion = np.zeros(shape=(classes, classes))

        for i, (train, test) in enumerate(self.cv):
            if flip:
                train, test = test, train

            temp = deepcopy(self.data)

            # scale then normalize a copy of data based on training set for fairness w.r.t. sapinet.
            if norm:
                temp[:] = normalize(temp[:].cpu(), p=1)

            if scale:
                # NOTE scaling factor(s) for the entire set are computed based on the training set.
                max_values, _ = torch.max(torch.abs(temp[train]), dim=0)
                temp[:] = torch.nan_to_num(temp[:] / max_values)

            # fit the model.
            svm.fit(temp[train], self.labels[train])

            # compute various scores (accuracy, MCC, F1, precision, recall).
            scores[i] = 100.0 * svm.score(temp[test], self.labels[test])
            mcc[i] = matthews_corrcoef(y_pred=svm.predict(temp[test]), y_true=self.labels[test])
            # extended.append(precision_recall_fscore_support(y_pred=svm.predict(temp[test]), y_true=self.labels[test]))

            # compute confusion matrix and add to accumulated count across folds.
            conf_mat = confusion_matrix(
                y_true=self.labels[test], y_pred=svm.predict(temp[test]), labels=np.unique(self.labels)
            )
            confusion += conf_mat

            # advance fold/iteration counts and append current fold data to file(s), if applicable.
            self.fold += 1

            if file:
                ensure_dir(os.path.dirname(file))
                self.save(
                    file + "_detail",
                    results={"Accuracy": scores[i], "MCC": mcc[i]},
                    confusion=None,
                    extended=None,
                )

        # scores was initialized to max possible length (LOO), but now we can trim to how many folds there were.
        scores = scores[: self.fold]
        mcc = mcc[: self.fold]

        acc_mean = np.mean(scores)
        mcc_mean = np.mean(mcc)

        acc_ci = stats.t.interval(ci_alpha, len(scores) - 1, loc=np.mean(scores), scale=stats.sem(scores))
        mcc_ci = stats.t.interval(ci_alpha, len(mcc) - 1, loc=np.mean(mcc), scale=stats.sem(mcc))

        self.fold = 0
        if file:
            self.save(
                file + "_group",
                results={
                    "Accuracy": acc_mean,
                    "Acc_CI0": acc_ci[0],
                    "Acc_CI1": acc_ci[1],
                    "MCC": mcc_mean,
                    "MCC_CI0": mcc_ci[0],
                    "MCC_CI1": mcc_ci[1],
                },
                confusion=confusion,
            )

        return acc_mean, acc_ci, confusion

    def grid_svm(self, kernel: str = "rbf", seed: int = 96) -> (dict[str, float], float):
        """Cross validated grid search optimization of C and gamma for a full dataset."""
        c_range = np.logspace(-7, 9, 17)
        gamma_range = np.logspace(-8, 10, 19)

        param_grid = dict(gamma=gamma_range, C=c_range)
        grid = GridSearchCV(
            SVC(kernel=kernel, class_weight="balanced", random_state=seed),
            param_grid=param_grid,
            cv=self.cv.cross_validator,
            n_jobs=-1,
            verbose=0,
        )

        grid.fit(self.data, self.labels)
        # print(f"Optimal: {grid.best_params_}")

        return grid.best_params_, grid.best_score_

    def save(self, file, results, confusion=None, extended=None):
        """Saves classification results and parameters to `file`.

        Confusion matrices and extended metrics (e.g., F1, precision, recall) are optional arguments.

        Note
        ----
        Opens the file in append mode by default. Caller should ensure the correct file is added to.

        """
        with open(file + ".csv", "a") as f:
            writer = csv.writer(f, delimiter=",")

            # header should include all identifiers deemed relevant by the caller.
            # those may include experiment, session, fold, model ID.
            if os.path.getsize(file + ".csv") == 0:
                header = list(self.identifiers) + ["InternalFold"] + list(results)
                writer.writerow(header)

            params = [str(p) for p in self.identifiers.values()] + [self.fold] + [str(p) for p in results.values()]
            writer.writerow(params)

        if extended is not None:
            extended = np.array(extended)

            with open(file + "_ext.csv", "a") as f:
                writer = csv.writer(f, delimiter=",")

                if os.path.getsize(file + "_ext.csv") == 0:
                    header = list(self.identifiers) + [
                        "InternalFold",
                        "Class",
                        "Precision",
                        "Recall",
                        "FScore",
                        "Support",
                    ]
                    writer.writerow(header)

                    for cls in range(len(extended[0])):
                        params = [str(p) for p in self.identifiers.values()] + [self.fold, cls + 1]
                        params.extend(extended[:, cls])

                        writer.writerow(params)

        if confusion is not None:
            np.savez(file + "_conf.npz", confusion)


def baseline_svm(self, data: SNNData, shots: int, key: str | list[str], file: str = None):
    """Fit a baseline SVM to the original dataset, assumed normalized."""
    # identifiers for logging.
    svm_id = {
        "Session": self.session,
        "Condition": self.condition,
        "Variant": self.variant,
        "Analysis": "SVM",
        "Mode": self.mode,
        "Layer": "RBF",
        "Batches": self.configuration.get("sampling", {}).get("included_batches", -1),
        "BatchesTr": self.configuration.get("sampling", {}).get("train_batches", -1),
        "BatchesTe": self.configuration.get("sampling", {}).get("test_batches", -1),
        "Shots": self.configuration.get("sampling", {}).get("shots", -1),
    }

    optimizer = SVM(
        identifiers=svm_id,
        data=data[: data[:].shape[0] // 2].cpu(),
        labels=data.metadata[key][: data[:].shape[0] // 2],
        cv=CV(
            data=data.trim(index=slice(0, data[:].shape[0] // 2)),
            cross_validator=LimitedStratifiedKFold(
                n_splits=self.configuration.get("sampling", {}).get("folds", 2) // 2,
            ),
            label_keys=key,
        ),
    )
    hyper, _ = optimizer.grid_svm(kernel="rbf", seed=self.seed)

    score, ci, confusion = SVM(
        identifiers=svm_id,
        data=data[data[:].shape[0] // 2 :].cpu(),
        labels=data.metadata[key][data[:].shape[0] // 2 :],
        cv=CV(
            data=data.trim(index=slice(data[:].shape[0] // 2, data[:].shape[0])),
            cross_validator=LimitedStratifiedKFold(
                n_splits=self.configuration.get("sampling", {}).get("folds", 2) // 2
            ),
            label_keys=key,
        ),
    ).score_svm(
        kernel="rbf",
        c=hyper["C"],
        gamma=hyper["gamma"],
        flip=True,
        scale=True,
        norm=True,
        seed=self.seed,
        file=file,
    )

    print(f"{shots}-shot SVM baseline: {score:.2f} % " f"(CI = {ci[0]:.2f} - {ci[1]:.2f})\n")
    return score, ci, confusion
