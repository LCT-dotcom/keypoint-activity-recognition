from __future__ import annotations

import copy
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterSampler

from tsfel_histgb_pipeline import WindowFeatures


@dataclass(frozen=True)
class ClassicalConfig:
    feature_budget: int
    learning_rate: float
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    weighting: str

    def to_dict(self) -> dict:
        return asdict(self)


class TrainingOnlyFeatureSelector:
    def __init__(
        self,
        feature_budget: int,
        random_state: int = 42,
        correlation_threshold: float = 0.98,
    ) -> None:
        if feature_budget <= 0:
            raise ValueError("feature_budget must be positive")
        if not 0 < correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be in (0, 1]")
        self.feature_budget = int(feature_budget)
        self.random_state = int(random_state)
        self.correlation_threshold = float(correlation_threshold)
        self.selected_columns: tuple[str, ...] = ()
        self.fill_values_: pd.Series | None = None
        self.mi_scores_: pd.Series | None = None
        self.constant_columns_: tuple[str, ...] = ()
        self.correlated_columns_: tuple[str, ...] = ()

    @staticmethod
    def _numeric_frame(x: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(x, pd.DataFrame) or x.empty:
            raise ValueError("x must be a non-empty DataFrame")
        if x.columns.duplicated().any():
            raise ValueError("x must have unique feature columns")
        return x.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def fit(self, x: pd.DataFrame, y: Sequence[str]) -> TrainingOnlyFeatureSelector:
        frame = self._numeric_frame(x)
        labels = np.asarray(list(y), dtype=object)
        if len(frame) != len(labels) or len(labels) == 0:
            raise ValueError("x and y must be non-empty and have equal row counts")

        fill_values = frame.median(axis=0, skipna=True).fillna(0.0)
        imputed = frame.fillna(fill_values)
        varying_mask = imputed.nunique(dropna=False).gt(1)
        constant_columns = imputed.columns[~varying_mask].tolist()
        candidates = imputed.loc[:, varying_mask]
        if candidates.empty:
            raise ValueError("No non-constant features remain after filtering")

        correlation = candidates.corr(method="spearman").abs()
        upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
        correlated_columns = [column for column in upper.columns if (upper[column] > self.correlation_threshold).any()]
        candidates = candidates.drop(columns=correlated_columns)
        if candidates.empty:
            raise ValueError("No features remain after correlation filtering")

        if len(np.unique(labels)) < 2:
            scores = np.zeros(candidates.shape[1], dtype=float)
        else:
            scores = mutual_info_classif(
                candidates.to_numpy(dtype=float),
                labels,
                discrete_features=False,
                random_state=self.random_state,
            )
        ranked = pd.DataFrame({"feature": candidates.columns, "score": scores})
        ranked = ranked.sort_values(["score", "feature"], ascending=[False, True], kind="mergesort")
        selected = ranked["feature"].head(min(self.feature_budget, len(ranked))).tolist()

        self.fill_values_ = fill_values
        self.mi_scores_ = pd.Series(scores, index=candidates.columns, name="mutual_information")
        self.constant_columns_ = tuple(constant_columns)
        self.correlated_columns_ = tuple(correlated_columns)
        self.selected_columns = tuple(selected)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_columns:
            raise ValueError("Feature selector has not been fitted")
        missing = [column for column in self.selected_columns if column not in x.columns]
        if missing:
            raise ValueError(f"Input is missing selected features: {missing}")
        frame = self._numeric_frame(x.loc[:, list(self.selected_columns)])
        return frame

    def fit_transform(self, x: pd.DataFrame, y: Sequence[str]) -> pd.DataFrame:
        return self.fit(x, y).transform(x)

    def restrict_budget(self, feature_budget: int) -> TrainingOnlyFeatureSelector:
        if not self.selected_columns:
            raise ValueError("Feature selector has not been fitted")
        if feature_budget <= 0:
            raise ValueError("feature_budget must be positive")
        restricted = copy.deepcopy(self)
        restricted.feature_budget = int(feature_budget)
        restricted.selected_columns = self.selected_columns[:feature_budget]
        return restricted


@dataclass
class FittedClassicalModel:
    config: ClassicalConfig
    selector: TrainingOnlyFeatureSelector
    imputer: SimpleImputer
    classifier: HistGradientBoostingClassifier

    def predict_proba(self, x: pd.DataFrame, class_order: Sequence[str]) -> np.ndarray:
        selected = self.selector.transform(x)
        transformed = self.imputer.transform(selected)
        fitted_probabilities = self.classifier.predict_proba(transformed)
        requested = tuple(class_order)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("class_order must be non-empty and unique")
        probabilities = np.zeros((len(selected), len(requested)), dtype=float)
        requested_index = {label: index for index, label in enumerate(requested)}
        unknown = [label for label in self.classifier.classes_ if label not in requested_index]
        if unknown:
            raise ValueError(f"class_order is missing fitted classes: {unknown}")
        for fitted_index, label in enumerate(self.classifier.classes_):
            probabilities[:, requested_index[label]] = fitted_probabilities[:, fitted_index]
        return probabilities

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        selected = self.selector.transform(x)
        return self.classifier.predict(self.imputer.transform(selected))


def _sample_weights(labels: pd.Series, weighting: str) -> np.ndarray | None:
    if weighting == "none":
        return None
    if weighting != "sqrt_balanced":
        raise ValueError(f"Unsupported weighting mode: {weighting}")
    counts = labels.value_counts()
    n_rows = len(labels)
    n_classes = len(counts)
    by_class = {label: np.sqrt(n_rows / (n_classes * count)) for label, count in counts.items()}
    return labels.map(by_class).to_numpy(dtype=float)


def fit_classical_model(
    windows: WindowFeatures,
    config: ClassicalConfig,
    random_state: int = 42,
    prefit_selector: TrainingOnlyFeatureSelector | None = None,
) -> FittedClassicalModel:
    if windows.x.empty or windows.y.empty or len(windows.x) != len(windows.y):
        raise ValueError("Training windows must be non-empty and aligned")
    if prefit_selector is None:
        selector = TrainingOnlyFeatureSelector(
            feature_budget=config.feature_budget,
            random_state=random_state,
        )
        selected = selector.fit_transform(windows.x, windows.y)
    else:
        if not prefit_selector.selected_columns:
            raise ValueError("prefit_selector has not been fitted")
        if prefit_selector.feature_budget != config.feature_budget:
            raise ValueError("prefit_selector budget does not match config")
        selector = prefit_selector
        selected = selector.transform(windows.x)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    transformed = imputer.fit_transform(selected)
    classifier = HistGradientBoostingClassifier(
        learning_rate=config.learning_rate,
        max_iter=250,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=20,
        random_state=random_state,
    )
    weights = _sample_weights(windows.y, config.weighting)
    classifier.fit(transformed, windows.y, sample_weight=weights)
    return FittedClassicalModel(
        config=config,
        selector=selector,
        imputer=imputer,
        classifier=classifier,
    )


def phase_d_parameter_candidates(random_state: int = 42) -> list[ClassicalConfig]:
    distributions = {
        "feature_budget": [128, 256, 512, 1024],
        "learning_rate": [0.03, 0.06, 0.10],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [20, 40, 80],
        "l2_regularization": [0.0, 1.0, 5.0],
        "weighting": ["none", "sqrt_balanced"],
    }
    sampled = ParameterSampler(distributions, n_iter=24, random_state=random_state)
    return [ClassicalConfig(**dict(parameters)) for parameters in sampled]


def build_inner_subject_splits(
    subject_ids: Sequence[int],
    outer_held_out_subject: int,
) -> list[tuple[tuple[int, ...], int]]:
    unique_subjects = tuple(sorted(set(int(subject) for subject in subject_ids)))
    if outer_held_out_subject not in unique_subjects:
        raise ValueError("outer_held_out_subject is not present")
    development_subjects = tuple(subject for subject in unique_subjects if subject != outer_held_out_subject)
    if len(development_subjects) < 2:
        raise ValueError("Nested subject selection requires at least three total subjects")
    return [
        (tuple(subject for subject in development_subjects if subject != validation), validation)
        for validation in development_subjects
    ]
