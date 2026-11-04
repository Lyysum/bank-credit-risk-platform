"""生成可复现的个人信贷申请合成数据，并注入可追踪的数据质量问题。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import RAW_DIR, SEED, DEFAULT_ROWS


def generate_applications(rows: int = DEFAULT_ROWS, seed: int = SEED) -> pd.DataFrame:
    if rows < 800:
        raise ValueError("至少生成 800 条记录，以保证训练和评估样本充足。")
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 70, rows)
    income = np.clip(rng.lognormal(np.log(82000), .55, rows), 18000, 500000).round(0)
    dti = np.clip(rng.beta(2.1, 5.2, rows) * .92, .01, .92).round(3)
    amount = np.clip(income * rng.uniform(.08, .9, rows), 3000, 250000).round(0)
    credit_years = np.clip(age - rng.integers(18, 31, rows), 0, 45)
    delin = rng.poisson(.2 + dti * 1.7, rows).clip(0, 6)
    employment_years = np.clip(rng.gamma(2.1, 3.1, rows), 0, 42).round(1)
    employment = rng.choice(["受薪", "自雇", "合同制"], rows, p=[.70, .18, .12])
    purpose = rng.choice(["消费", "购车", "装修", "教育", "债务整合"], rows, p=[.34, .18, .20, .10, .18])
    home = rng.choice(["自有", "按揭", "租房"], rows, p=[.27, .43, .30])
    channel = rng.choice(["手机银行", "网点", "合作渠道", "开放平台"], rows, p=[.50, .20, .20, .10])
    region = rng.choice(["广州", "深圳", "佛山", "东莞", "珠海", "其他"], rows,
                        p=[.26, .28, .16, .14, .06, .10])
    month_idx = rng.integers(0, 18, rows)
    months = pd.Period("2025-01", freq="M") + month_idx
    application_month = months.astype(str)
    logit = (-3.0 + 3.7 * dti + .46 * delin + .60 * (employment == "自雇")
             + .43 * (home == "租房") + .40 * (amount / np.maximum(income, 1) > .58)
             - .016 * credit_years - .010 * employment_years
             + .22 * (channel == "合作渠道"))
    probability = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, probability)
    frame = pd.DataFrame({
        "application_id": [f"APP{i:08d}" for i in range(1, rows + 1)],
        "application_month": application_month,
        "age": age, "annual_income": income, "loan_amount": amount,
        "debt_to_income": dti, "credit_history_years": credit_years,
        "delinquencies_2y": delin, "employment_years": employment_years,
        "employment_type": employment, "loan_purpose": purpose,
        "home_ownership": home, "channel": channel, "region": region,
        "default": default,
    })
    # Inject realistic, documented defects: missingness, duplicate keys, invalid amounts.
    for col in ["annual_income", "employment_years", "employment_type"]:
        ix = rng.choice(rows, max(1, rows // 100), replace=False)
        frame.loc[ix, col] = np.nan
    dup = frame.sample(frac=.008, random_state=seed).copy()
    bad = frame.sample(frac=.004, random_state=seed + 1).index
    frame.loc[bad, "loan_amount"] = -frame.loc[bad, "loan_amount"]
    raw = pd.concat([frame, dup], ignore_index=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_DIR / "applications_raw.csv", index=False)
    return raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = generate_applications(args.rows, args.seed)
    print(f"原始数据生成完成：{len(result):,} 行，文件：data/raw/applications_raw.csv")

