"""Reproducible local credit-risk demo pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
NUMERIC = ["age", "annual_income", "loan_amount", "debt_to_income", "credit_history_years", "delinquencies_2y", "employment_years"]
CATEGORICAL = ["employment_type", "loan_purpose", "home_ownership"]
FEATURES = NUMERIC + CATEGORICAL


def make_data(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 70, rows)
    income = np.clip(rng.lognormal(np.log(78000), .48, rows), 18000, 350000).round(0)
    dti = np.clip(rng.beta(2.2, 5.5, rows) * .85, 0, .85).round(3)
    amount = np.clip(income * rng.uniform(.08, .75, rows), 3000, 180000).round(0)
    credit_years = np.clip(age - rng.integers(18, 30, rows), 0, 45)
    delin = rng.poisson(.25 + dti * 1.6, rows).clip(0, 5)
    emp = np.clip(rng.gamma(2.0, 3.0, rows), 0, 40).round(1)
    employment = rng.choice(["受薪", "自雇", "合同制"], rows, p=[.72, .16, .12])
    purpose = rng.choice(["消费", "购车", "装修", "教育", "债务整合"], rows, p=[.35, .18, .2, .1, .17])
    home = rng.choice(["自有", "按揭", "租房"], rows, p=[.28, .42, .3])
    logit = -3.15 + 4.0 * dti + .48 * delin + .65 * (employment == "自雇") + .48 * (home == "租房") + .35 * (amount / np.maximum(income, 1) > .55) - .018 * credit_years - .012 * emp
    probability = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, probability)
    frame = pd.DataFrame({"application_id": [f"APP{i:07d}" for i in range(1, rows + 1)], "age": age, "annual_income": income,
        "loan_amount": amount, "debt_to_income": dti, "credit_history_years": credit_years,
        "delinquencies_2y": delin, "employment_years": emp, "employment_type": employment,
        "loan_purpose": purpose, "home_ownership": home, "default": default})
    for col in ["annual_income", "employment_years", "employment_type"]:
        frame.loc[rng.choice(rows, max(1, rows // 100), replace=False), col] = np.nan
    return frame


def validate(frame: pd.DataFrame) -> dict:
    required = {"application_id", *FEATURES, "default"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"缺少必要字段: {missing_columns}")
    duplicate_ids = int(frame.application_id.duplicated().sum())
    invalid_labels = int((~frame.default.isin([0, 1])).sum())
    if duplicate_ids or invalid_labels:
        raise ValueError(f"数据校验失败: duplicate_ids={duplicate_ids}, invalid_labels={invalid_labels}")
    return {"rows": len(frame), "duplicate_application_ids": duplicate_ids,
            "invalid_labels": invalid_labels, "missing_cells": int(frame[FEATURES].isna().sum().sum()),
            "default_rate": round(float(frame.default.mean()), 4)}


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    return ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])


def run(rows: int = 5000, seed: int = 42) -> dict:
    if rows < 500:
        raise ValueError("样本数至少为 500，便于进行分层训练/测试切分。")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    data = make_data(rows, seed)
    quality = validate(data)
    data.to_csv(PROCESSED / "credit_applications.csv", index=False)
    X_train, X_test, y_train, y_test = train_test_split(data[FEATURES], data.default, test_size=.25,
        random_state=seed, stratify=data.default)
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=4, class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
    }
    metrics, fitted = {}, {}
    for name, estimator in candidates.items():
        model = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = model.predict_proba(X_test)[:, 1]
        metrics[name] = {"roc_auc": round(float(roc_auc_score(y_test, score)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4)}
        fitted[name] = model
    selected = max(metrics, key=lambda name: metrics[name]["roc_auc"])
    joblib.dump(fitted[selected], MODELS / "credit_risk_model.joblib")
    scores = data[["application_id", *FEATURES, "default"]].copy()
    scores["risk_score"] = fitted[selected].predict_proba(data[FEATURES])[:, 1].round(6)
    scores["risk_band"] = pd.cut(scores.risk_score, [-.001, .25, .5, 1.001], labels=["低", "中", "高"]).astype(str)
    scores.to_csv(PROCESSED / "risk_scores.csv", index=False)
    report = {"data_quality": quality, "model_metrics": metrics, "selected_model": selected,
        "threshold": .5, "split": {"method": "stratified_random", "test_size": .25, "seed": seed},
        "notice": "仅为合成数据教学演示，不可用于真实授信决策。"}
    (PROCESSED / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行银行个人信贷风险演示流水线")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.rows, args.seed), ensure_ascii=False, indent=2))
