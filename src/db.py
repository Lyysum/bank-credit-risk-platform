"""SQLite 本地数仓连接与分层表读写。"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from config import DB_PATH


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def write_table(frame: pd.DataFrame, table: str, if_exists: str = "replace") -> None:
    with connect() as conn:
        frame.to_sql(table, conn, if_exists=if_exists, index=False)


def read_table(table: str, path: Path = DB_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"数仓不存在：{path}，请先运行 python run_all.py")
    with connect(path) as conn:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', conn)


def query(sql: str, params: tuple = (), path: Path = DB_PATH) -> pd.DataFrame:
    with connect(path) as conn:
        return pd.read_sql_query(sql, conn, params=params)

