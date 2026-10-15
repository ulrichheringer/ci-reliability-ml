"""CI failure-cause classification and flaky-test risk estimation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SEED = 42
CAUSES = {
    "dependency": ["dependency package version could not be resolved", "dependency lockfile is out of date", "dependency download returned 503", "dependency module not found during install", "build uses incompatible dependency library version", "dependency registry authentication failed"],
    "timeout": ["job timeout exceeded maximum duration", "request timeout waiting for service", "test process timed out after startup", "connection timeout deadline exceeded", "runner killed task after timeout", "integration timeout check did not finish"],
    "assertion": ["assertion expected response status 200 but got 500", "assertion failed for calculated total", "assertion snapshot differs from expected output", "test assertion expected a nonempty collection", "assertion value differs from expected result", "unit test assertion comparison failed"],
    "environment": ["environment container image could not start", "environment variable missing in runner", "environment permission denied creating temporary file", "environment service port is already in use", "environment runner has insufficient disk space", "system environment package is unavailable"],
}
FEATURES = ["historical_fail_rate", "duration_cv", "concurrent_runs", "network_calls", "retry_count", "environment_count"]


def _cause_data(seed=SEED):
    texts, labels, groups = [], [], []
    for label, templates in CAUSES.items():
        for template_id, template in enumerate(templates):
            for run in range(12):
                texts.append(f"{template}; job {run}; runner {('linux', 'windows', 'macos')[run % 3]}")
                labels.append(label)
                groups.append(f"{label}-{template_id}")
    return texts, labels, groups


def _flaky_data(seed=SEED, rows=1000):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "historical_fail_rate": rng.beta(1.5, 5.0, rows),
        "duration_cv": rng.gamma(1.3, 0.45, rows),
        "concurrent_runs": rng.integers(1, 9, rows),
        "network_calls": rng.integers(0, 12, rows),
        "retry_count": rng.integers(0, 4, rows),
        "environment_count": rng.integers(1, 5, rows),
    })
    score = (2.2 * X.historical_fail_rate + 0.75 * X.duration_cv + 0.08 * X.concurrent_runs
             + 0.07 * X.network_calls + 0.16 * X.retry_count + 0.08 * X.environment_count
             + rng.normal(0, 0.2, rows))
    y = (score > 1.45).astype(int)
    return X, y


def sample_input():
    return {
        "failure_log": "dependency download returned 503 while resolving package versions",
        "flaky_features": {"historical_fail_rate": 0.22, "duration_cv": 0.8, "concurrent_runs": 6,
                           "network_calls": 8, "retry_count": 2, "environment_count": 3},
    }


def train_model(seed=SEED):
    texts, labels, _ = _cause_data(seed)
    cause_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LinearSVC(C=1.0, random_state=seed)).fit(texts, labels)
    X, y = _flaky_data(seed)
    flaky_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)).fit(X, y)
    return {"cause": cause_model, "flaky": flaky_model}


def predict(model, payload):
    row = pd.DataFrame([payload["flaky_features"]], columns=FEATURES)
    return {
        "failure_cause": str(model["cause"].predict([payload["failure_log"]])[0]),
        "flaky_risk": bool(model["flaky"].predict(row)[0]),
        "flaky_risk_probability": round(float(model["flaky"].predict_proba(row)[0, 1]), 4),
    }


def evaluate(seed=SEED):
    texts, labels, groups = _cause_data(seed)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(texts, labels, groups))
    cause_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LinearSVC(C=1.0, random_state=seed)).fit([texts[i] for i in tr], [labels[i] for i in tr])
    cause_pred = cause_model.predict([texts[i] for i in te])
    X, y = _flaky_data(seed)
    x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    flaky_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)).fit(x_tr, y_tr)
    flaky_pred = flaky_model.predict(x_te)
    return {
        "cause_accuracy": round(float(accuracy_score([labels[i] for i in te], cause_pred)), 4),
        "cause_macro_f1": round(float(f1_score([labels[i] for i in te], cause_pred, average="macro")), 4),
        "flaky_accuracy": round(float(accuracy_score(y_te, flaky_pred)), 4),
        "flaky_f1": round(float(f1_score(y_te, flaky_pred)), 4),
        "flaky_roc_auc": round(float(roc_auc_score(y_te, flaky_model.predict_proba(x_te)[:, 1])), 4),
        "cause_test_rows": int(len(te)), "flaky_test_rows": int(len(y_te)),
    }
