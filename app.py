"""交互式银行个人信贷风险分析看板。"""
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DB_PATH, OUTPUT_DIR
from src.db import read_table

st.set_page_config(page_title="银行信贷风险分析平台", page_icon="🏦", layout="wide")
st.title("🏦 银行个人信贷风险分析与预警平台")
st.caption("ODS → DWD → DWS → ADS · 时间外验证 · 合成数据教学演示，不用于真实授信")

report_path = OUTPUT_DIR / "model_report.json"
quality_path = OUTPUT_DIR / "data_quality.json"
if not DB_PATH.exists() or not report_path.exists():
    st.warning("尚未生成数仓和模型结果。先在项目目录运行 `python run_all.py`。")
    st.stop()

report = json.loads(report_path.read_text(encoding="utf-8"))
quality = json.loads(quality_path.read_text(encoding="utf-8"))
scores = read_table("ads_risk_scores")
scores["application_month"] = pd.to_datetime(scores.application_month)
with st.sidebar:
    st.header("筛选条件")
    regions = st.multiselect("地区", sorted(scores.region.unique()), default=sorted(scores.region.unique()))
    channels = st.multiselect("申请渠道", sorted(scores.channel.unique()), default=sorted(scores.channel.unique()))
    min_month, max_month = scores.application_month.min().date(), scores.application_month.max().date()
    date_range = st.date_input("申请月份范围", (min_month, max_month), min_value=min_month, max_value=max_month)

filtered = scores[scores.region.isin(regions) & scores.channel.isin(channels)].copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.offsets.MonthEnd(0)
    filtered = filtered[filtered.application_month.between(start, end)]
if filtered.empty:
    st.info("当前筛选条件下没有记录。")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("申请量", f"{len(filtered):,}")
k2.metric("合成违约率", f"{filtered.default.mean():.1%}")
k3.metric("高风险占比", f"{filtered.risk_band.eq('高').mean():.1%}")
k4.metric("平均贷款额", f"¥{filtered.loan_amount.mean():,.0f}")
k5.metric("平均负债收入比", f"{filtered.debt_to_income.mean():.1%}")

tab1, tab2, tab3, tab4 = st.tabs(["组合总览", "客群与渠道", "模型评估", "数据质量"])
with tab1:
    left, right = st.columns(2)
    monthly = filtered.assign(month=filtered.application_month.dt.strftime("%Y-%m")).groupby("month", as_index=False).agg(
        申请量=("application_id", "count"), 违约率=("default", "mean"))
    with left:
        st.subheader("月度申请量与违约率")
        fig = go.Figure()
        fig.add_bar(x=monthly.month, y=monthly.申请量, name="申请量", opacity=.65)
        fig.add_scatter(x=monthly.month, y=monthly.违约率, name="违约率", yaxis="y2", mode="lines+markers")
        fig.update_layout(yaxis2={"overlaying": "y", "side": "right", "tickformat": ".0%"}, legend={"orientation": "h"})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("风险评分分布")
        st.plotly_chart(px.histogram(filtered, x="risk_score", color="risk_band", nbins=35,
            labels={"risk_score": "预测违约概率", "risk_band": "风险等级"}), use_container_width=True)
    band = filtered.groupby("risk_band", as_index=False, observed=False).agg(
        申请量=("application_id", "count"), 实际违约率=("default", "mean"))
    st.subheader("风险分层表现")
    st.dataframe(band, use_container_width=True, hide_index=True)
with tab2:
    left, right = st.columns(2)
    with left:
        st.subheader("地区风险对比")
        region_kpi = filtered.groupby("region", as_index=False).agg(申请量=("application_id", "count"),
            违约率=("default", "mean"), 平均负债收入比=("debt_to_income", "mean"))
        st.plotly_chart(px.bar(region_kpi, x="region", y="违约率", color="申请量", text_auto=".1%"), use_container_width=True)
    with right:
        st.subheader("申请渠道风险对比")
        channel_kpi = filtered.groupby("channel", as_index=False).agg(申请量=("application_id", "count"), 违约率=("default", "mean"))
        st.plotly_chart(px.bar(channel_kpi, x="channel", y="违约率", color="申请量", text_auto=".1%"), use_container_width=True)
    purpose = filtered.groupby("loan_purpose", as_index=False).agg(申请量=("application_id", "count"), 违约率=("default", "mean"))
    st.subheader("贷款用途分析")
    st.plotly_chart(px.scatter(purpose, x="申请量", y="违约率", size="申请量", color="loan_purpose", text="loan_purpose"), use_container_width=True)
with tab3:
    st.subheader("时间外测试集模型指标")
    st.caption(f"最终模型：{report['selected_model']}；决策阈值：{report['decision_threshold']:.2f}；测试月份：{', '.join(report['split']['test_months'])}")
    metrics = pd.DataFrame([report["test_metrics"]]).T.rename(columns={0: "测试集指标"})
    st.dataframe(metrics, use_container_width=True)
    st.subheader("验证集模型比较")
    st.dataframe(pd.DataFrame(report["validation_model_comparison"]).T, use_container_width=True)
    importance = read_table("ads_model_feature_importance").head(15)
    st.subheader("模型特征贡献（模型内置重要性）")
    st.plotly_chart(px.bar(importance.sort_values("importance"), x="importance", y="feature", orientation="h"), use_container_width=True)
    st.warning("风险分数是合成数据上的模型输出；不能据此批准或拒绝真实贷款。")
with tab4:
    st.subheader("数据治理结果")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("ODS 原始记录", f"{quality['raw_rows']:,}")
    q2.metric("去除重复", f"{quality['duplicate_rows_removed']:,}")
    q3.metric("隔离异常", f"{quality['invalid_rows_quarantined']:,}")
    q4.metric("DWD 有效记录", f"{quality['dwd_rows']:,}")
    st.write("空值分布")
    st.dataframe(pd.DataFrame.from_dict(quality["missing_by_column"], orient="index", columns=["缺失数量"]), use_container_width=True)
    st.caption(f"质量通过率：{quality['quality_pass_rate']:.1%}。异常记录保留在 `dwd_rejected_applications` 隔离表中。")

st.subheader("高风险申请（当前筛选结果）")
st.dataframe(filtered[filtered.risk_band == "高"].sort_values("risk_score", ascending=False).head(100),
              use_container_width=True, hide_index=True)

