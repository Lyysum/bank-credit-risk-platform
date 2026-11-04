"""项目参数集中管理。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"
DB_PATH = ROOT / "data" / "warehouse" / "credit_risk.db"
MODEL_DIR = ROOT / "models"
SEED = 42
DEFAULT_ROWS = 12000
TEST_SIZE = 0.25


