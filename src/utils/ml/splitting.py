""" Custom cross validation splitters. """
import warnings
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_random_state, column_or_1d, indexable

__all__ = ("LimitedStratifiedKFold",)


class LimitedStratifiedKFold(StratifiedKFold):
    """A StratifiedKFold variant that allows for limiting train/test set selection to a subset of the indices
    (e.g., based on group identity) while retaining full flexibility w.r.t. sampling and number of folds.

    The `tr_mask` parameter is for the segment intended to be smaller (in few-shot scenarios, that would be the
    training set, hence the naming convention). The `te_mask` is for ensuring only specific indices will be tested,
    as opposed to everything that isn't in the training fold.

    If neither mask is supplied, behaves like StratifiedKFold.

    Example
    -------
    >>> data = np.array([0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10, 11, 12, 13, 14])
    >>> labels = np.array(["a", "b", "b", "b", "a", "a", "a", "a", "b", "b", "b", "a", "b", "a"])
    >>> batches = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3])

    >>> cv_obj = LimitedStratifiedKFold(n_splits=3, tr_mask=(groups == 1), te_mask=(groups != 3), shuffle=True)

    >>> for te, tra in cv_obj.split(X, y, groups=groups):
    >>>    print(f"Train: {tra}, Test: {te}")

    """

    def __init__(self, n_splits=5, *, tr_mask=None, te_mask=None, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        self.tr_mask = tr_mask
        self.te_mask = te_mask

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(len(X))

        for test_index in self._iter_test_masks(X, y, groups):
            if self.te_mask is None:
                retain = np.logical_not(test_index)
            else:
                retain = np.logical_not(test_index) * np.array(self.te_mask)

            train_index = indices[retain]
            test_index = indices[test_index]
            yield train_index, test_index

    def _make_test_folds(self, X, y=None):
        """Overrides original implementation."""
        rng = check_random_state(self.random_state)

        y = np.asarray(y)
        if self.tr_mask is not None:
            y = y[self.tr_mask]

        type_of_target_y = type_of_target(y)
        allowed_target_types = ("binary", "multiclass")
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                "Supported target types are: {}. Got {!r} instead.".format(allowed_target_types, type_of_target_y)
            )

        y = column_or_1d(y)

        _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = class_perm[y_inv]

        n_classes = len(y_idx)
        y_counts = np.bincount(y_encoded)
        min_groups = np.min(y_counts)
        if np.all(self.n_splits > y_counts):
            raise ValueError(
                "n_splits=%d cannot be greater than the" " number of members in each class." % self.n_splits
            )
        if self.n_splits > min_groups:
            warnings.warn(
                "The least populated class in y has only %d"
                " members, which is less than n_splits=%d." % (min_groups, self.n_splits),
                UserWarning,
            )

        y_order = np.sort(y_encoded)
        allocation = np.asarray(
            [
                # get every n_splits-th element starting at i.
                np.bincount(y_order[i :: self.n_splits], minlength=n_classes)
                for i in range(self.n_splits)
            ]
        )

        test_folds = np.empty(len(y), dtype="i")
        for k in range(n_classes):
            folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
            if self.shuffle:
                rng.shuffle(folds_for_class)
            test_folds[y_encoded == k] = folds_for_class

        if self.tr_mask is None:
            return test_folds

        corrected = np.zeros_like(self.tr_mask) - 1
        j = 0
        for i, index in enumerate(self.tr_mask):
            if self.tr_mask[i]:
                corrected[i] = test_folds[j]
                j += 1

        return corrected.reshape(corrected.shape[0])


if __name__ == "__main__":
    X = np.array([0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10, 11, 12, 13, 14])
    y = np.array(["a", "b", "b", "b", "a", "a", "a", "a", "b", "b", "b", "a", "b", "a"])
    groups = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3])

    cv = LimitedStratifiedKFold(n_splits=3, tr_mask=(groups == 1), te_mask=(groups != 3), shuffle=True)

    for test, train in cv.split(X, y, groups=groups):
        print(f"Train: {train}, Test: {test}")
