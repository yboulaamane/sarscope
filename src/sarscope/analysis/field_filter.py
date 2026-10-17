"""Training-only field screening without quadratic correlation allocation."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

from sarscope.params import FieldQsarParams


class FieldColumnFilter(BaseEstimator, TransformerMixin):
    """Training-only screening with an explicit missing-descriptor contract.

    A new CoMFA molecule overlapping a retained electrostatic grid location
    is unscorable in strict mode. Opt-in training_mean follows the SYBYL 6.6
    manual, QSAR §4.3.6: use the fitting-column mean, never validation/test
    statistics. All-missing columns are dropped and replaced terms are counted.

    Columns are ranked by standard deviation. Under ``field_screen="per_field"``
    each block is ranked only against itself and receives its own share of the
    budget, so the retained set is invariant to rescaling any single field.
    ``global_raw`` ranks all blocks together, which is only meaningful when they
    share units; see ``FieldQsarParams.field_screen``.

    After fitting, ``retained_per_field_`` gives the kept count per block under
    per-field screening (None otherwise) and ``empty_fields_`` lists any block
    indices that retained nothing.
    """

    #: Kept columns per field block under per-field screening; None otherwise.
    retained_per_field_: tuple[int, ...] | None
    #: Indices of blocks that retained no columns.
    empty_fields_: tuple[int, ...]

    def __init__(self, params: FieldQsarParams) -> None:
        self.params = params

    def fit(self, X: NDArray[Any], y: Any = None) -> FieldColumnFilter:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) < 2:
            raise ValueError("Field filtering needs at least two training rows.")
        self.n_features_in_ = X.shape[1]
        if np.isinf(X).any():
            raise ValueError(
                "Infinite fields are numerical errors, not steric-overlap missing data."
            )
        finite = np.isfinite(X)
        observed_count = finite.sum(axis=0)
        self.column_mean_ = np.sum(np.where(finite, X, 0.0), axis=0) / np.maximum(observed_count, 1)
        self.column_mean_[observed_count == 0] = np.nan
        self.n_undefined_ = int((~finite.all(axis=0)).sum())
        self.n_all_missing_ = int((observed_count == 0).sum())
        mean_policy = self.params.missing_field_policy == "training_mean"
        defined = observed_count > 0 if mean_policy else finite.all(axis=0)
        fitting_values = np.where(finite, X, self.column_mean_) if mean_policy else X
        sd = np.zeros(X.shape[1])
        sd[defined] = fitting_values[:, defined].std(axis=0, ddof=1)
        eligible = np.flatnonzero(defined & (sd > 1e-12))
        n_fields = len(self.params.field_subset) or (5 if self.params.method == "comsia" else 2)
        if self.params.field_screen == "per_field" and n_fields > 1:
            if X.shape[1] % n_fields:
                raise ValueError(
                    f"Per-field screening expects {n_fields} equal blocks, but "
                    f"{X.shape[1]} columns do not divide by {n_fields}. Check that "
                    "field_subset matches the matrix, or use field_screen='global_raw'."
                )
            width = X.shape[1] // n_fields
            # Each block gets its own share of the budget and is ranked only
            # against itself, so a block cannot be excluded for carrying larger
            # units than its neighbour.
            share = max(1, self.params.max_features // n_fields)
            blocks = []
            for index in range(n_fields):
                block = eligible[(eligible >= index * width) & (eligible < (index + 1) * width)]
                blocks.append(self._rank(block, sd, share))
            candidates = np.concatenate(blocks) if blocks else np.array([], dtype=int)
            self.retained_per_field_ = tuple(len(block) for block in blocks)
        else:
            candidates = self._rank(eligible, sd, self.params.max_features)
            self.retained_per_field_ = None
        candidates = np.sort(candidates)
        if not len(candidates):
            raise ValueError("No defined, varying fields survive training-only filtering.")
        empty = (
            []
            if self.retained_per_field_ is None
            else [index for index, count in enumerate(self.retained_per_field_) if count == 0]
        )
        self.empty_fields_ = tuple(empty)
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[candidates] = True
        self.n_after_variance_ = self.n_after_correlation_ = int(len(candidates))
        self.training_sd_ = sd
        self.column_scale_ = np.ones(X.shape[1])
        if self.params.field_scaling == "column_std":
            self.column_scale_[self.support_] = sd[self.support_]
        if X.shape[1] % n_fields and self.params.field_scaling != "raw":
            raise ValueError("Field blocks must have equal grid widths for block scaling.")
        if self.params.field_scaling == "block_std":
            width = X.shape[1] // n_fields
            for start in range(0, X.shape[1], width):
                block = np.arange(start, start + width)
                retained = block[self.support_[block]]
                if len(retained):
                    scale = float(fitting_values[:, retained].std(ddof=0))
                    self.column_scale_[block] = scale if scale > 1e-12 else 1.0
        return self

    def _rank(self, columns: NDArray[Any], sd: NDArray[Any], budget: int) -> NDArray[Any]:
        """Keep the highest-variance columns of one candidate set, within budget."""
        if not len(columns):
            return columns
        if self.params.filter_mode == "absolute":
            columns = columns[sd[columns] >= self.params.minimum_sd]
        else:
            count = max(1, int(np.ceil(len(columns) * self.params.keep_fraction)))
            columns = columns[np.argsort(-sd[columns], kind="stable")[:count]]
        return columns[np.argsort(-sd[columns], kind="stable")[:budget]]

    def transform(self, X: NDArray[Any]) -> NDArray[np.float64]:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Field matrix width differs from the training grid.")
        selected = X[:, self.support_]
        if np.isinf(selected).any():
            raise ValueError("Infinite fields cannot be replaced by training means.")
        if not np.isfinite(selected).all():
            if self.params.missing_field_policy == "training_mean":
                selected = np.where(np.isnan(selected), self.column_mean_[self.support_], selected)
            else:
                raise ValueError(
                    "CoMFA electrostatic overlap in held-out molecules: retained fields are "
                    "undefined. This split is not scorable; use CoMSIA or reviewed alignments. "
                    "Alternatively, explicitly set 'Missing electrostatic descriptors' to "
                    "'fitting-column means' and review the reported replacement fractions. "
                    "No test-derived mask or imputation was applied."
                )
        return selected / self.column_scale_[self.support_]

    def missing_counts(self, X: NDArray[Any]) -> NDArray[np.int64]:
        """Number of retained descriptor terms absent in each row; no fitted state changes."""
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("Field matrix width differs from the training grid.")
        if np.isinf(values[:, self.support_]).any():
            raise ValueError("Infinite fields cannot be counted as steric-overlap missing data.")
        return np.isnan(values[:, self.support_]).sum(axis=1).astype(np.int64)
