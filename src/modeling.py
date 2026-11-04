"""时间切分、基线比较、阈值选择和全量申请风险评分。"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import MODEL_DIR, OUTPUT_DIR, SEED
from src.db import write_table
from src.etl import CATEGORICAL, FEATURES, NUMERIC


def _pipeline(estimator):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    category = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("encode", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", category, CATEGORICAL)])
    return Pipeline([("preprocess", prep), ("model", estimator)])


def _metrics(y, probability, threshold):
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"roc_auc": round(float(roc_auc_score(y, probability)), 4),
        "pr_auc": round(float(average_precision_score(y, probability)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "brier_score": round(float(brier_score_loss(y, probability)), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def train_and_score(dwd: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    data = dwd.sort_values("application_month").reset_index(drop=True)
    months = sorted(data.application_month.unique())
    if len(months) < 6:
        raise ValueError("至少需要 6 个月数据才能进行训练、验证和测试的时间切分。")
    train_end = max(1, int(len(months) * .60))
    valid_end = max(train_end + 1, int(len(months) * .80))
    train_months, valid_months = months[:train_end], months[train_end:valid_end]
    test_months = months[valid_end:]
    if not test_months:
        test_months = months[-1:]
        valid_months = months[train_end:-1]
    train = data[data.application_month.isin(train_months)]
    valid = data[data.application_month.isin(valid_months)]
    test = data[data.application_month.isin(test_months)]
    if min(train.default.nunique(), valid.default.nunique(), test.default.nunique()) < 2:
        raise ValueError("时间切分后某个集合只有一个标签类别，请增加样本量或重新生成数据。")
    models = {
        "logistic_regression": _pipeline(LogisticRegression(max_iter=1200, class_weight="balanced", random_state=SEED)),
        "random_forest": _pipeline(RandomForestClassifier(n_estimators=240, min_samples_leaf=8,
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
    }
    comparison = {}
    for name, model in models.items():
        model.fit(train[FEATURES], train.default)
        comparison[name] = {"validation_roc_auc": round(float(roc_auc_score(
            valid.default, model.predict_proba(valid[FEATURES])[:, 1])), 4)}
    selected = max(comparison, key=lambda name: comparison[name]["validation_roc_auc"])
    model = models[selected]
    val_scores = model.predict_proba(valid[FEATURES])[:, 1]
    thresholds = np.arange(.10, .81, .01)
    threshold = float(max(thresholds, key=lambda t: f1_score(valid.default, val_scores >= t, zero_division=0)))
    test_scores = model.predict_proba(test[FEATURES])[:, 1]
    test_metrics = _metrics(test.default, test_scores, threshold)
    all_scores = model.predict_proba(data[FEATURES])[:, 1]
    scored = data[["application_id", "application_month", "region", "channel", "loan_purpose",
                   "annual_income", "loan_amount", "debt_to_income", "default"]].copy()
    scored["risk_score"] = all_scores.round(6)
    scored["risk_band"] = pd.cut(scored.risk_score, [-.001, .25, .5, 1.001],
                                  labels=["低", "中", "高"]).astype(str)
    write_table(scored, "ads_risk_scores")
    scored.to_csv(OUTPUT_DIR / "risk_scores.csv", index=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "credit_risk_model.joblib")
    prep = model.named_steps["preprocess"]
    names = prep.get_feature_names_out()
    estimator = model.named_steps["model"]
    raw_importance = (np.abs(estimator.coef_[0]) if hasattr(estimator, "coef_")
                      else estimator.feature_importances_)
    feature_importance = pd.DataFrame({"feature": names, "importance": raw_importance})
    feature_importance = feature_importance.sort_values("importance", ascending=False).head(20)
    write_table(feature_importance, "ads_model_feature_importance")
    report = {"selected_model": selected, "validation_model_comparison": comparison,
        "test_metrics": test_metrics, "decision_threshold": round(threshold, 2),
        "split": {"method": "chronological_train_validation_test", "train_months": [str(x) for x in train_months],
                  "validation_months": [str(x) for x in valid_months], "test_months": [str(x) for x in test_months]},
        "notice": "所有输入均为合成数据；指标只用于方法演示，不可用于真实授信。"}
    (OUTPUT_DIR / "model_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return scored, report

