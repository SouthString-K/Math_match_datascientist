"""
Task 4: 市场短期动态响应模拟（伪三周回测）
data_loader — 事件 / 公司画像 / 关联矩阵 / 股价历史加载
"""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ---------- 路径配置 ----------
EVENTS_PATH = Path(r"D:\Math_match\codes\outputs\classification_run_eastmoney400_v2\classification_results_new.json")
COMPANY_PROFILES_PATH = Path(r"D:\Math_match\codes\outputs\task2_profile_run\company_profiles.json")
ASSOC_MATRIX_PATH = Path(r"D:\Math_match\codes\outputs\task2_attribute_correction\assoc_matrix.csv")
STOCK_HISTORY_PATHS = [
    Path(r"D:\Math_match\codes\task3\hs300_history_batch1.json"),
    Path(r"D:\Math_match\codes\task3\hs300_history_batch2.json"),
]

# ---------- 字段名映射 ----------
PRICE_FIELDS_EN = [
    "open", "close", "high", "low",
    "volume", "amount", "amplitude",
    "pct_change", "change_amount", "turnover_rate",
]
PRICE_FIELDS_CN = [
    "开盘", "收盘", "最高", "最低",
    "成交量", "成交额", "振幅",
    "涨跌幅", "涨跌额", "换手率",
]

# ---------- 辅助函数 ----------
def normalize_stock_code(code) -> str:
    return str(code).strip().lstrip("0").zfill(6)


def safe_float(value, default=0.0) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, str) and not value.strip():
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------- 事件加载 ----------
def load_events() -> list:
    """返回事件列表，每条含 sample_id / publish_time / event特征"""
    events = json.loads(EVENTS_PATH.read_text(encoding="utf-8"))
    result = []
    for e in events:
        result.append({
            "sample_id": e["sample_id"],
            "publish_time": e.get("publish_time", ""),   # "4.11" ~ "4.14"
            "event_type": e.get("event_type", ""),
            "duration_type": e.get("duration_type", ""),
            "heat": safe_float(e.get("heat", 0)),
            "event_intensity": safe_float(e.get("event_intensity", 0)),
            "influence_range": safe_float(e.get("influence_range", 0)),
            "attribute_score": safe_float(e.get("attribute_score", 0)),
            "event_summary": e.get("event_summary", ""),
            "title": e.get("title", ""),
        })
    return result


def parse_event_date(publish_time: str, year: int = 2026) -> str:
    """'4.11' → '20260411'"""
    if not publish_time:
        return ""
    try:
        month, day = publish_time.strip().split(".")
        return f"{year}{int(month):02d}{int(day):02d}"
    except Exception:
        return ""


# ---------- 公司画像加载 ----------
def load_company_lookup():
    """返回 {stock_code: company_dict}"""
    payload = json.loads(COMPANY_PROFILES_PATH.read_text(encoding="utf-8"))
    companies = payload.get("companies", payload) if isinstance(payload, dict) else payload
    return {normalize_stock_code(c["stock_code"]): c for c in companies}


# ---------- 关联矩阵加载 ----------
def load_assoc_matrix() -> pd.DataFrame:
    """返回 DataFrame: index=stock_code(int), columns=event_id"""
    df = pd.read_csv(ASSOC_MATRIX_PATH, encoding="utf-8-sig", index_col=0)
    # index 转为字符串 stock_code
    df.index = df.index.astype(str).str.zfill(6)
    df.columns = df.columns.astype(str)
    return df


# ---------- 股价历史加载 ----------
def load_stock_history():
    """返回 {stock_code: {date: row_dict}}"""
    merged = defaultdict(dict)
    for path in STOCK_HISTORY_PATHS:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for stock_code, rows in payload.get("data_by_stock", {}).items():
            norm = normalize_stock_code(stock_code)
            for row in rows:
                date = str(row.get("日期") or row.get("date", ""))
                if not date:
                    continue
                normalized = {
                    "date": date,
                    "stock_code": norm,
                    "open": safe_float(row.get("开盘") or row.get("open")),
                    "close": safe_float(row.get("收盘") or row.get("close")),
                    "high": safe_float(row.get("最高") or row.get("high")),
                    "low": safe_float(row.get("最低") or row.get("low")),
                    "volume": safe_float(row.get("成交量") or row.get("volume")),
                    "amount": safe_float(row.get("成交额") or row.get("amount")),
                    "amplitude": safe_float(row.get("振幅") or row.get("amplitude")),
                    "pct_change": safe_float(row.get("涨跌幅") or row.get("pct_change")),
                    "change_amount": safe_float(row.get("涨跌额") or row.get("change_amount")),
                    "turnover_rate": safe_float(row.get("换手率") or row.get("turnover_rate")),
                }
                merged[norm][date] = normalized

    # 按日期排序
    sorted_history = {}
    for code, by_date in merged.items():
        dates = sorted(by_date.keys())
        sorted_history[code] = [by_date[d] for d in dates]
    return sorted_history


def get_stock_row(history: dict, stock_code: str, date: str):
    """获取指定日期的行情行，不存在则返回 None"""
    code_history = history.get(normalize_stock_code(stock_code), [])
    for row in code_history:
        if row["date"] == date:
            return row
    return None


def get_trading_date_before(history: dict, stock_code: str, ref_date: str) -> str:
    """获取 ref_date 之前最近一个有数据的交易日"""
    code_history = history.get(normalize_stock_code(stock_code), [])
    dates = [r["date"] for r in code_history]
    earlier = [d for d in dates if d < ref_date]
    return max(earlier) if earlier else ""


# ---------- 窗口划分 ----------
def build_windows():
    """
    伪三周实验窗口：
    窗口1: 4.11 事件集
    窗口2: 4.12–4.13 事件集
    窗口3: 4.14 事件集
    """
    return [
        {"window_id": 1, "publish_times": {"4.11"}, "buy_date": "20260415", "sell_date": "20260417"},
        {"window_id": 2, "publish_times": {"4.12", "4.13"}, "buy_date": "20260416", "sell_date": "20260418"},
        {"window_id": 3, "publish_times": {"4.14"}, "buy_date": "20260417", "sell_date": "20260421"},
    ]


# ---------- 事件筛选（Task4 Step 1）----------
EVENT_FILTER = {
    "heat_min": 0.7,
    "intensity_min": 0.7,
    "range_min": 0.7,
    "attribute_min": 0.8,
}


def filter_events(events: list, publish_times: set) -> list:
    """按阈值筛选事件"""
    filtered = []
    for e in events:
        if str(e.get("publish_time", "")) not in publish_times:
            continue
        if e["heat"] < EVENT_FILTER["heat_min"]:
            continue
        if e["event_intensity"] < EVENT_FILTER["intensity_min"]:
            continue
        if e["influence_range"] < EVENT_FILTER["range_min"]:
            continue
        if e["attribute_score"] < EVENT_FILTER["attribute_min"]:
            continue
        filtered.append(e)
    return filtered


# ---------- 公司池构建（Task4 Step 2）----------
def build_stock_pool(events: list, assoc_matrix: pd.DataFrame, company_lookup: dict) -> list:
    """
    对每个事件取 Top-30 关联公司，合并去重
    筛选条件：match_priority='高' 且 confidence >= 0.8
    """
    PRIORITY_MAP = {"高": 1.0, "中": 0.5, "低": 0.0}
    seen = set()
    pool = []

    for evt in events:
        sid = evt["sample_id"]
        if sid not in assoc_matrix.columns:
            continue
        scores = assoc_matrix[sid].sort_values(ascending=False)
        top30 = scores.head(30)

        for code, assoc_val in top30.items():
            code_str = str(code).zfill(6)
            if code_str in seen:
                continue
            seen.add(code_str)

            company = company_lookup.get(code_str)
            if company is None:
                continue
            priority_raw = company.get("match_priority", "低")
            priority = PRIORITY_MAP.get(priority_raw, 0.0) if isinstance(priority_raw, str) else safe_float(priority_raw)
            if priority < 1.0:   # 只保留 match_priority='高'
                continue
            conf = safe_float(company.get("confidence", 0))
            if conf < 0.8:
                continue

            # 事件敏感类型匹配
            sens_types = company.get("event_sensitive_types", []) or []
            evt_type = evt.get("event_type", "")
            if sens_types and evt_type not in sens_types and evt_type:
                continue

            pool.append({
                "stock_code": code_str,
                "stock_name": company.get("stock_name", code_str),
                "assoc_ij": float(assoc_val),
                "match_priority": priority_raw,
                "confidence": conf,
                "industry_lv1": company.get("industry_lv1", ""),
            })

    return pool


if __name__ == "__main__":
    print("[DataLoader] 测试加载...")
    events = load_events()
    print(f"  事件总数: {len(events)}")

    assoc = load_assoc_matrix()
    print(f"  关联矩阵: {assoc.shape}")

    history = load_stock_history()
    print(f"  股票历史: {len(history)} 只股票")

    windows = build_windows()
    for w in windows:
        filtered = filter_events(events, w["publish_times"])
        pool = build_stock_pool(filtered, assoc, load_company_lookup())
        print(f"  窗口{w['window_id']}: {len(filtered)} 事件, {len(pool)} 候选股票")
