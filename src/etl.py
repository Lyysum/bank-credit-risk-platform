"""ODS→DWD→DWS 数据加工及数据质量治理。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import OUTPUT_DIR
from src.db import write_table

NUMERIC = ["age", "annual_income", "loan_amount", "debt_to_income",
           "credit_history_years", "delinquencies_2y", "employment_years"]
CATEGORICAL = ["employment_type", "loan_purpose", "home_ownership", "channel", "region"]
FEATURES = NUMERIC + CATEGORICAL


def run_etl(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    write_table(raw, "ods_applications")
    n_raw = len(raw)
    deduped = raw.drop_duplicates("application_id", keep="first").copy()
    duplicate_count = n_raw - len(deduped)
    valid = (deduped["age"].between(18, 100)
             & deduped["annual_income"].between(0, 2_000_000)
             & deduped["loan_amount"].between(0, 1_000_000)
             & deduped["debt_to_income"].between(0, 1)
             & deduped["default"].isin([0, 1]))
    rejected = deduped.loc[~valid].copy()
    clean = deduped.loc[valid].copy()
    clean["application_month"] = pd.to_datetime(clean["application_month"] + "-01")
    clean["loan_to_income"] = (clean["loan_amount"] / clean["annual_income"].replace(0, np.nan)).clip(0, 20)
    clean["income_band"] = pd.cut(clean["annual_income"],
        bins=[-1, 40000, 80000, 150000, np.inf], labels=["低于4万", "4-8万", "8-15万", "15万以上"])
    clean["dti_band"] = pd.cut(clean["debt_to_income"],
        bins=[-0.001, .2, .4, .6, 1.001], labels=["低", "中低", "中高", "高"])
    write_table(clean, "dwd_applications")
    write_table(rejected, "dwd_rejected_applications")
    by_month = clean.groupby("application_month", as_index=False).agg(
        application_count=("application_id", "count"), default_count=("default", "sum"),
        default_rate=("default", "mean"), avg_loan_amount=("loan_amount", "mean"))
    by_channel = clean.groupby("channel", as_index=False).agg(
        application_count=("application_id", "count"), default_count=("default", "sum"),
        default_rate=("default", "mean"), avg_dti=("debt_to_income", "mean"))
    by_region = clean.groupby("region", as_index=False).agg(
        application_count=("application_id", "count"), default_count=("default", "sum"),
        default_rate=("default", "mean"), avg_income=("annual_income", "mean"))
    for table, frame in [("dws_monthly_risk", by_month), ("dws_channel_risk", by_channel),
                         ("dws_region_risk", by_region)]:
        write_table(frame, table)
    quality = {
        "raw_rows": n_raw, "duplicate_rows_removed": duplicate_count,
        "invalid_rows_quarantined": len(rejected), "dwd_rows": len(clean),
        "missing_cells_after_rules": int(clean[FEATURES].isna().sum().sum()),
        "missing_by_column": clean[FEATURES].isna().sum().loc[lambda x: x.gt(0)].to_dict(),
        "default_rate": round(float(clean.default.mean()), 4),
        "quality_pass_rate": round(len(clean) / max(n_raw, 1), 4),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "data_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    return clean, quality

