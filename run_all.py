"""一键执行：合成数据→ODS/DWD/DWS/ADS→质量治理→分析→模型评分。"""
import argparse
import json

from config import DEFAULT_ROWS, SEED
from src.analysis import analyze_portfolio
from src.data_generator import generate_applications
from src.etl import run_etl
from src.modeling import train_and_score


def main(rows: int = DEFAULT_ROWS, seed: int = SEED) -> None:
    raw = generate_applications(rows, seed)
    dwd, quality = run_etl(raw)
    kpis = analyze_portfolio(dwd)
    _, model_report = train_and_score(dwd)
    print(json.dumps({"data_quality": quality, "portfolio_kpis": kpis,
                      "model_report": model_report}, ensure_ascii=False, indent=2))
    print("\n完整流水线运行完成。看板启动命令：streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行本地银行信贷风险端到端项目")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    main(args.rows, args.seed)

