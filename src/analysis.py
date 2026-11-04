"""信贷组合、渠道、区域、客群和月份维度分析。"""
from __future__ import annotations

import json

import pandas as pd

from config import OUTPUT_DIR
from src.db import write_table


def analyze_portfolio(dwd: pd.DataFrame) -> dict:
    base = dwd.copy()
    base["application_month"] = pd.to_datetime(base.application_month).dt.strftime("%Y-%m")
    segment_specs = [("region", "ads_region_kpi"), ("channel", "ads_channel_kpi"),
                     ("loan_purpose", "ads_purpose_kpi"), ("income_band", "ads_income_kpi"),
                     ("dti_band", "ads_dti_kpi")]
    for dim, table in segment_specs:
        result = base.groupby(dim, observed=False, dropna=False).agg(
            application_count=("application_id", "count"), default_count=("default", "sum"),
            default_rate=("default", "mean"), avg_loan_amount=("loan_amount", "mean"),
            avg_dti=("debt_to_income", "mean")).reset_index()
        write_table(result, table)
    monthly = base.groupby("application_month", as_index=False).agg(
        application_count=("application_id", "count"), default_count=("default", "sum"),
        default_rate=("default", "mean"), avg_loan_amount=("loan_amount", "mean"))
    write_table(monthly, "ads_monthly_kpi")
    total = len(base)
    kpis = {"application_count": total, "default_count": int(base.default.sum()),
        "default_rate": round(float(base.default.mean()), 4),
        "avg_loan_amount": round(float(base.loan_amount.mean()), 2),
        "avg_annual_income": round(float(base.annual_income.mean()), 2),
        "avg_dti": round(float(base.debt_to_income.mean()), 4),
        "high_risk_dti_share": round(float(base.debt_to_income.ge(.6).mean()), 4)}
    (OUTPUT_DIR / "portfolio_kpis.json").write_text(json.dumps(kpis, ensure_ascii=False, indent=2), encoding="utf-8")
    return kpis

