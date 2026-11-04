from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "data" / "processed" / "report.json"
SCORES = ROOT / "data" / "processed" / "risk_scores.csv"

st.set_page_config(page_title="个人信贷风险分析", page_icon="🏦", layout="wide")
st.title("🏦 个人信贷风险分析与预警")
st.caption("本地教学原型 · 合成数据 · 不用于真实授信决策")

if not REPORT.exists() or not SCORES.exists():
    st.info("尚无流水线结果。请先在项目目录运行：`python -m src.pipeline --rows 5000`，然后刷新本页。")
    st.stop()

report = json.loads(REPORT.read_text(encoding="utf-8"))
df = pd.read_csv(SCORES)
high = df[df.risk_band == "高"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("申请样本", f"{len(df):,}")
c2.metric("合成违约率", f"{df.default.mean():.1%}")
c3.metric("高风险样本", f"{len(high):,}")
c4.metric("当前模型", report["selected_model"])

left, right = st.columns(2)
with left:
    st.subheader("风险分数分布")
    st.plotly_chart(px.histogram(df, x="risk_score", color="risk_band", nbins=30,
        labels={"risk_score": "预测违约风险分数", "risk_band": "风险分层"}), use_container_width=True)
with right:
    st.subheader("风险分层与实际标签")
    summary = df.groupby("risk_band", observed=False).agg(申请数=("application_id", "count"),
        合成违约率=("default", "mean")).reset_index()
    st.plotly_chart(px.bar(summary, x="risk_band", y="申请数", color="合成违约率",
        labels={"risk_band": "风险分层"}), use_container_width=True)

st.subheader("模型评估（测试集）")
st.dataframe(pd.DataFrame(report["model_metrics"]).T, use_container_width=True)
st.subheader("高风险样本（按分数排序）")
st.dataframe(high.sort_values("risk_score", ascending=False).head(100), use_container_width=True)
