"""
Task 4 Step 4: 持仓与卖出策略
- 资金分配：w_i = Score_i / Σ Score_k
- 买入：开盘价买入
- 卖出：收盘价卖出
- 收益率 = Profit / Capital
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

INITIAL_CAPITAL = 100_000.0   # 起始资金 10 万

WINDOW_TRADE_PATH = Path(r"D:\Math_match\codes\task3\dataset\window_stock_data.json")


def load_window_trade_prices():
    """加载预计算的交易价格 {window_id: {stock_code: {buy_open, sell_close}}}"""
    raw = json.loads(WINDOW_TRADE_PATH.read_text(encoding="utf-8"))
    result = defaultdict(dict)
    for item in raw:
        result[item["window_id"]][item["stock_code"]] = {
            "buy_open": float(item["buy_open_price"]),
            "sell_close": float(item["sell_close_price"]),
            "buy_date": item["buy_date"],
            "sell_date": item["sell_date"],
            "stock_name": item["stock_name"],
        }
    return result


def load_trade_prices(stock_codes: list, buy_date: str, sell_date: str, history: dict) -> dict:
    """
    获取股票池中所有股票在买卖两日的开盘/收盘价
    返回 {stock_code: {buy_open, sell_close, buy_date, sell_date}}
    """
    prices = {}
    for code in stock_codes:
        buy_row = None
        sell_row = None
        for day in history.get(code, []):
            d = day["date"]
            if d == buy_date:
                buy_row = day
            elif d == sell_date:
                sell_row = day

        if buy_row and sell_row:
            prices[code] = {
                "buy_open": float(buy_row["open"]),
                "sell_close": float(sell_row["close"]),
                "buy_date": buy_date,
                "sell_date": sell_date,
            }
    return prices


def simulate_window(
    window_id: int,
    selected_stocks: list,
    prices: dict,
    capital: float,
) -> dict:
    """
    模拟单窗口交易

    Args:
        window_id: 窗口编号
        selected_stocks: fusion 输出的选中股票列表（含 weight）
        prices: {stock_code: {buy_open, sell_close, ...}}
        capital: 当前可用资金
        history: 全量股价历史

    Returns:
        dict: {window_id, capital_before, capital_after, profit, return_rate, positions}
    """
    # 分配资金
    positions = []
    for stock in selected_stocks:
        code = stock["stock_code"]
        price_info = prices.get(code)
        if price_info is None:
            continue

        buy_open = price_info["buy_open"]
        if buy_open <= 0:
            continue

        allocated = capital * stock["weight"]
        shares = int(allocated / buy_open)
        cost = shares * buy_open
        remaining = allocated - cost

        sell_close = price_info["sell_close"]
        revenue = shares * sell_close
        profit = revenue - cost

        positions.append({
            "stock_code": code,
            "stock_name": stock["stock_name"],
            "weight": stock["weight"],
            "allocated": round(allocated, 2),
            "buy_open": buy_open,
            "shares": shares,
            "cost": round(cost, 2),
            "sell_close": sell_close,
            "revenue": round(revenue, 2),
            "profit": round(profit, 2),
        })

    total_profit = sum(p["profit"] for p in positions)
    total_cost = sum(p["cost"] for p in positions)
    return_rate = total_profit / capital if capital > 0 else 0.0

    return {
        "window_id": window_id,
        "capital_before": round(capital, 2),
        "total_allocated": round(sum(p["allocated"] for p in positions), 2),
        "total_cost": round(total_cost, 2),
        "profit": round(total_profit, 2),
        "return_rate": round(return_rate, 6),
        "capital_after": round(capital + total_profit, 2),
        "n_positions": len(positions),
        "positions": positions,
    }


def simulate_trading(
    selected_stocks_by_window: dict,
    windows: list,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """
    完整三窗口模拟
    windows: build_windows() 的输出
    selected_stocks_by_window: {window_id: [selected stocks list]}（来自 fusion）
    """
    window_prices = load_window_trade_prices()

    capital = initial_capital
    window_results = []
    capital_history = []

    for window in windows:
        wid = window["window_id"]
        buy_date = window["buy_date"]
        sell_date = window["sell_date"]

        selected = selected_stocks_by_window.get(wid, [])
        if not selected:
            print(f"[Trading] 窗口{wid}: 无选中股票，跳过", flush=True)
            window_results.append({
                "window_id": wid,
                "profit": 0.0,
                "return_rate": 0.0,
                "capital_after": round(capital, 2),
                "positions": [],
            })
            continue

        # 使用预计算的交易价格
        prices_by_code = window_prices.get(wid, {})

        # 构建 simulate_window 所需的 prices dict（仅保留在预计算价格中的股票）
        matched_stocks = []
        for stock in selected:
            code = stock["stock_code"]
            if code in prices_by_code:
                matched_stocks.append(stock)

        if not matched_stocks:
            print(f"[Trading] 窗口{wid}: 选中股票无交易价格，跳过", flush=True)
            window_results.append({
                "window_id": wid,
                "profit": 0.0,
                "return_rate": 0.0,
                "capital_after": round(capital, 2),
                "positions": [],
            })
            continue

        # 重新归一化权重（只考虑有价格的股票）
        total_score = sum(s["score"] for s in matched_stocks if s.get("score", 0) > 0)
        for s in matched_stocks:
            s["weight"] = round(s["score"] / total_score, 4) if total_score > 0 else 0.0

        result = simulate_window(wid, matched_stocks, prices_by_code, capital)

        capital = result["capital_after"]
        capital_history.append({
            "window_id": wid,
            "capital_before": result["capital_before"],
            "capital_after": capital,
            "profit": result["profit"],
            "return_rate": result["return_rate"],
        })

        window_results.append(result)

        print(f"[Trading] === 窗口{wid} ===", flush=True)
        print(f"  买入日: {buy_date}  卖出日: {sell_date}", flush=True)
        print(f"  资金: {result['capital_before']:,.2f} → {result['capital_after']:,.2f}", flush=True)
        print(f"  收益: {result['profit']:,.2f}  收益率: {result['return_rate']:.4%}", flush=True)
        for pos in result["positions"]:
            print(f"  {pos['stock_code']} {pos['stock_name']}  "
                  f"买{pos['buy_open']}×{pos['shares']}股  "
                  f"卖{pos['sell_close']}  "
                  f"盈利{pos['profit']:,.2f}", flush=True)
        print(flush=True)

    # 最终统计
    total_profit = sum(w["profit"] for w in window_results)
    final_return = total_profit / initial_capital

    # 夏普比率（简化：用窗口收益标准差估算）
    returns = [w["return_rate"] for w in window_results]
    mean_ret = np.mean(returns) if returns else 0.0
    std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
    sharpe = (mean_ret / std_ret) if std_ret > 0 else 0.0

    # 最大回撤
    peak = initial_capital
    max_drawdown = 0.0
    for ch in capital_history:
        peak = max(peak, ch["capital_after"])
        dd = (peak - ch["capital_after"]) / peak if peak > 0 else 0.0
        max_drawdown = max(max_drawdown, dd)

    summary = {
        "initial_capital": initial_capital,
        "final_capital": round(capital, 2),
        "total_profit": round(total_profit, 2),
        "total_return_rate": round(final_return, 6),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_drawdown, 6),
        "n_windows": len(windows),
        "n_profitable": sum(1 for w in window_results if w["profit"] > 0),
        "window_results": window_results,
        "capital_history": capital_history,
    }

    return summary
