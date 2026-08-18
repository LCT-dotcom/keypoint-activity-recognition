from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from phase_d_classical import (
    ClassicalConfig,
    TrainingOnlyFeatureSelector,
    build_inner_subject_splits,
    fit_classical_model,
    phase_d_parameter_candidates,
)
from phase_d_evaluation import FrameProbabilityResult, aggregate_window_probabilities, evaluate_frame_result
from tsfel_histgb_pipeline import (
    CLASSES,
    WindowFeatures,
    concatenate_window_features,
    evaluate_predictions,
    fit_estimator,
)


@dataclass
class CandidateResult:
    name: str
    fold_metrics: pd.DataFrame
    pooled_accuracy: float
    pooled_macro_f1: float
    pooled_abnormal_f1: float
    is_baseline: bool = False
    eligible: bool = False
    eligibility_reasons: list[str] = field(default_factory=list)

    @property
    def worst_subject_accuracy(self) -> float:
        if self.fold_metrics.empty or "accuracy" not in self.fold_metrics:
            raise ValueError(f"Candidate {self.name} has no fold accuracies")
        return float(self.fold_metrics["accuracy"].min())

    @property
    def subject_accuracies(self) -> dict[int, float]:
        required = {"held_out_subject", "accuracy"}
        if not required <= set(self.fold_metrics.columns):
            raise ValueError(f"Candidate {self.name} fold metrics are missing {sorted(required)}")
        return {
            int(row.held_out_subject): float(row.accuracy)
            for row in self.fold_metrics[["held_out_subject", "accuracy"]].itertuples(index=False)
        }


@dataclass
class OuterFoldResult:
    candidate_name: str
    held_out_subject: int
    metrics: dict[str, float]
    frame_result: FrameProbabilityResult
    y_true: np.ndarray
    y_pred: np.ndarray
    confusion: pd.DataFrame
    classification_report: pd.DataFrame
    audit: dict
    fitted_model: object | None = None


@dataclass
class ClassicalSelection:
    config: ClassicalConfig
    inner_metrics: pd.DataFrame
    audit: dict


def validate_outer_audit(audit: dict) -> None:
    required = {"outer_held_out_subject", "outer_train_subjects", "inner_splits"}
    missing = sorted(required - set(audit))
    if missing:
        raise ValueError(f"Audit is missing fields: {missing}")
    held_out = int(audit["outer_held_out_subject"])
    outer_train = {int(subject) for subject in audit["outer_train_subjects"]}
    if held_out in outer_train:
        raise ValueError("Outer held-out subject leaked into outer training subjects")
    for split in audit["inner_splits"]:
        inner_train = {int(subject) for subject in split["train_subjects"]}
        validation = int(split["validation_subject"])
        if held_out in inner_train or validation == held_out:
            raise ValueError("Outer held-out subject leaked into inner split")
        if validation in inner_train:
            raise ValueError("Inner validation subject leaked into inner training subjects")
        if not inner_train <= outer_train or validation not in outer_train:
            raise ValueError("Inner split contains a subject outside outer training subjects")


def select_phase_d_winner(
    candidates: Sequence[CandidateResult],
    abnormal_tolerance: float = 0.005,
    max_subject_loss: float = 0.05,
) -> CandidateResult:
    baselines = [candidate for candidate in candidates if candidate.is_baseline]
    if len(baselines) != 1:
        raise ValueError("Exactly one baseline candidate is required")
    baseline = baselines[0]
    baseline_subjects = baseline.subject_accuracies
    baseline.eligible = True
    baseline.eligibility_reasons = []

    for candidate in candidates:
        if candidate is baseline:
            continue
        reasons: list[str] = []
        if candidate.pooled_abnormal_f1 < baseline.pooled_abnormal_f1 - abnormal_tolerance:
            reasons.append("abnormal_f1_below_tolerance")
        if candidate.worst_subject_accuracy < baseline.worst_subject_accuracy:
            reasons.append("worst_subject_accuracy_below_baseline")
        candidate_subjects = candidate.subject_accuracies
        if set(candidate_subjects) != set(baseline_subjects):
            reasons.append("subject_set_mismatch")
        else:
            for subject, baseline_accuracy in baseline_subjects.items():
                if candidate_subjects[subject] < baseline_accuracy - max_subject_loss:
                    reasons.append(f"subject_{subject}_loss_exceeds_limit")
        candidate.eligible = not reasons
        candidate.eligibility_reasons = reasons

    eligible = [candidate for candidate in candidates if candidate.eligible]
    return max(
        eligible,
        key=lambda candidate: (
            candidate.pooled_accuracy,
            candidate.worst_subject_accuracy,
            candidate.pooled_macro_f1,
            candidate.pooled_abnormal_f1,
        ),
    )


def _align_sklearn_probabilities(model, x: pd.DataFrame) -> np.ndarray:
    fitted = model.predict_proba(x)
    probabilities = np.zeros((len(x), len(CLASSES)), dtype=float)
    class_to_index = {label: index for index, label in enumerate(CLASSES)}
    for fitted_index, label in enumerate(model.classes_):
        if label not in class_to_index:
            raise ValueError(f"Model produced an unknown class: {label}")
        probabilities[:, class_to_index[label]] = fitted[:, fitted_index]
    return probabilities


def _build_outer_result(
    candidate_name: str,
    held_out_subject: int,
    frame_result: FrameProbabilityResult,
    truth: Sequence[str],
    audit: dict,
    fitted_model: object,
) -> OuterFoldResult:
    truth_array = np.asarray(truth, dtype=object)
    valid = np.asarray([label in set(CLASSES) for label in truth_array], dtype=bool)
    metrics_with_report = evaluate_frame_result(truth_array, frame_result, valid_classes=CLASSES)
    metrics = {key: value for key, value in metrics_with_report.items() if key != "classification_report"}
    y_true = truth_array[valid]
    y_pred = frame_result.labels[valid]
    confusion = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=CLASSES),
        index=CLASSES,
        columns=CLASSES,
    )
    report = pd.DataFrame(
        classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    ).transpose()
    validate_outer_audit(audit)
    return OuterFoldResult(
        candidate_name=candidate_name,
        held_out_subject=held_out_subject,
        metrics=metrics,
        frame_result=frame_result,
        y_true=y_true,
        y_pred=y_pred,
        confusion=confusion,
        classification_report=report,
        audit=audit,
        fitted_model=fitted_model,
    )


def run_d0_fold(
    subject_windows: dict[int, WindowFeatures],
    frame_truth: dict[int, Sequence[str]],
    held_out_subject: int,
    random_state: int = 42,
) -> OuterFoldResult:
    subject_ids = sorted(subject_windows)
    if set(subject_ids) != set(frame_truth):
        raise ValueError("Window and frame-truth subject sets must match")
    if held_out_subject not in subject_windows:
        raise ValueError("Held-out subject is unavailable")
    train_subjects = [subject for subject in subject_ids if subject != held_out_subject]
    train = concatenate_window_features([subject_windows[subject] for subject in train_subjects])
    test = subject_windows[held_out_subject]
    model = fit_estimator(train, random_state=random_state + held_out_subject)
    window_probabilities = _align_sklearn_probabilities(model, test.x)
    frame_result = aggregate_window_probabilities(
        test.meta,
        window_probabilities,
        n_frames=len(frame_truth[held_out_subject]),
        classes=CLASSES,
    )
    audit = {
        "outer_held_out_subject": held_out_subject,
        "outer_train_subjects": train_subjects,
        "inner_splits": [],
        "random_state": random_state,
        "candidate": "D0-C",
    }
    return _build_outer_result(
        "D0-C",
        held_out_subject,
        frame_result,
        frame_truth[held_out_subject],
        audit,
        model,
    )


def select_classical_config(
    subject_windows: dict[int, WindowFeatures],
    frame_truth: dict[int, Sequence[str]],
    outer_held_out_subject: int,
    candidates: Sequence[ClassicalConfig] | None = None,
    random_state: int = 42,
    selector_cache: dict | None = None,
    model_cache: dict | None = None,
) -> ClassicalSelection:
    subject_ids = sorted(subject_windows)
    if set(subject_ids) != set(frame_truth):
        raise ValueError("Window and frame-truth subject sets must match")
    splits = build_inner_subject_splits(subject_ids, outer_held_out_subject)
    candidate_list = list(candidates or phase_d_parameter_candidates(random_state))
    if not candidate_list:
        raise ValueError("At least one classical candidate is required")
    if selector_cache is None:
        selector_cache = {}
    if model_cache is None:
        model_cache = {}

    max_budget = max(config.feature_budget for config in candidate_list)
    split_data: list[tuple[tuple[int, ...], int, WindowFeatures, WindowFeatures, TrainingOnlyFeatureSelector]] = []
    selector_cache_hits = 0
    for inner_index, (train_subjects, validation_subject) in enumerate(splits):
        train = concatenate_window_features([subject_windows[subject] for subject in train_subjects])
        validation = subject_windows[validation_subject]
        selector_key = (tuple(train_subjects), max_budget, random_state)
        if selector_key in selector_cache:
            selector = selector_cache[selector_key]
            selector_cache_hits += 1
        else:
            selector = TrainingOnlyFeatureSelector(
                feature_budget=max_budget,
                random_state=random_state + sum(train_subjects),
            ).fit(train.x, train.y)
            selector_cache[selector_key] = selector
        split_data.append((train_subjects, validation_subject, train, validation, selector))

    rows: list[dict] = []
    model_cache_hits = 0
    for candidate_index, config in enumerate(candidate_list):
        for _, (train_subjects, validation_subject, train, validation, full_selector) in enumerate(split_data):
            selector = full_selector.restrict_budget(config.feature_budget)
            model_key = (tuple(train_subjects), config, random_state)
            if model_key in model_cache:
                fitted = model_cache[model_key]
                model_cache_hits += 1
            else:
                fitted = fit_classical_model(
                    train,
                    config=config,
                    random_state=random_state + candidate_index + sum(train_subjects) * 100,
                    prefit_selector=selector,
                )
                model_cache[model_key] = fitted
            probabilities = fitted.predict_proba(validation.x, CLASSES)
            frame_result = aggregate_window_probabilities(
                validation.meta,
                probabilities,
                n_frames=len(frame_truth[validation_subject]),
                classes=CLASSES,
            )
            truth = np.asarray(frame_truth[validation_subject], dtype=object)
            valid = np.asarray([label in set(CLASSES) for label in truth], dtype=bool)
            evaluated = evaluate_predictions(truth[valid], frame_result.labels[valid])
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "validation_subject": validation_subject,
                    "accuracy": evaluated["accuracy"],
                    "macro_f1": evaluated["macro_f1"],
                    "abnormal_f1": evaluated["abnormal_f1"],
                }
            )

    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("candidate_index", as_index=False).agg(
        mean_accuracy=("accuracy", "mean"),
        worst_accuracy=("accuracy", "min"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_abnormal_f1=("abnormal_f1", "mean"),
    )
    best_index = int(
        summary.sort_values(
            ["mean_accuracy", "worst_accuracy", "mean_macro_f1", "mean_abnormal_f1"],
            ascending=False,
            kind="mergesort",
        ).iloc[0]["candidate_index"]
    )
    inner_splits = [
        {"train_subjects": list(train_subjects), "validation_subject": validation_subject}
        for train_subjects, validation_subject in splits
    ]
    audit = {
        "outer_held_out_subject": outer_held_out_subject,
        "outer_train_subjects": [subject for subject in subject_ids if subject != outer_held_out_subject],
        "inner_splits": inner_splits,
        "random_state": random_state,
        "selected_candidate_index": best_index,
        "selector_fit_count": len(split_data) - selector_cache_hits,
        "selector_cache_hits": selector_cache_hits,
        "model_cache_hits": model_cache_hits,
    }
    validate_outer_audit(audit)
    return ClassicalSelection(
        config=candidate_list[best_index],
        inner_metrics=metrics,
        audit=audit,
    )


def run_d1_fold(
    subject_windows: dict[int, WindowFeatures],
    frame_truth: dict[int, Sequence[str]],
    held_out_subject: int,
    candidates: Sequence[ClassicalConfig] | None = None,
    random_state: int = 42,
    selector_cache: dict | None = None,
    model_cache: dict | None = None,
) -> OuterFoldResult:
    selection = select_classical_config(
        subject_windows,
        frame_truth,
        outer_held_out_subject=held_out_subject,
        candidates=candidates,
        random_state=random_state,
        selector_cache=selector_cache,
        model_cache=model_cache,
    )
    return run_d1_fold_with_selection(
        subject_windows,
        frame_truth,
        held_out_subject=held_out_subject,
        config=selection.config,
        selection_audit=selection.audit,
        random_state=random_state,
    )


def run_d1_fold_with_selection(
    subject_windows: dict[int, WindowFeatures],
    frame_truth: dict[int, Sequence[str]],
    held_out_subject: int,
    config: ClassicalConfig,
    selection_audit: dict,
    random_state: int = 42,
) -> OuterFoldResult:
    if set(subject_windows) != set(frame_truth):
        raise ValueError("Window and frame-truth subject sets must match")
    if int(selection_audit.get("outer_held_out_subject", -1)) != held_out_subject:
        raise ValueError("Selection audit does not match the requested outer held-out subject")
    validate_outer_audit(selection_audit)
    train_subjects = sorted(subject for subject in subject_windows if subject != held_out_subject)
    if train_subjects != sorted(int(subject) for subject in selection_audit["outer_train_subjects"]):
        raise ValueError("Selection audit outer training subjects do not match available data")
    train = concatenate_window_features([subject_windows[subject] for subject in train_subjects])
    test = subject_windows[held_out_subject]
    selector = TrainingOnlyFeatureSelector(
        feature_budget=config.feature_budget,
        random_state=random_state + held_out_subject,
    ).fit(train.x, train.y)
    fitted = fit_classical_model(
        train,
        config=config,
        random_state=random_state + held_out_subject,
        prefit_selector=selector,
    )
    probabilities = fitted.predict_proba(test.x, CLASSES)
    frame_result = aggregate_window_probabilities(
        test.meta,
        probabilities,
        n_frames=len(frame_truth[held_out_subject]),
        classes=CLASSES,
    )
    audit = dict(selection_audit)
    audit.update(
        {
            "candidate": "D1-regularized-C",
            "selected_config": config.to_dict(),
        }
    )
    return _build_outer_result(
        "D1-regularized-C",
        held_out_subject,
        frame_result,
        frame_truth[held_out_subject],
        audit,
        fitted,
    )
