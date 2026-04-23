"""
Task 4 入口脚本：伪三周回测完整流程
1. 加载数据
2. 窗口事件筛选 + 公司池构建
3. Task3 推理样本构造
4. Task3 模型推理（或 mock）
5. 融合评分 + 选股
6. 交易模拟 + 收益计算
7. 保存所有中间文件
"""
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ---------- 项目路径 ----------
PROJECT_DIR = Path(r"D:\Math_match\codes")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "task4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 步骤1: 数据加载 ----------
print("=" * 60, flush=True)
print("Step 1: 加载数据", flush=True)
print("=" * 60, flush=True)

from data_loader import (
    load_events, load_company_lookup, load_assoc_matrix, load_stock_history,
    build_windows, filter_events, build_stock_pool,
)

events = load_events()
print(f"  事件总数: {len(events)}", flush=True)

assoc_matrix = load_assoc_matrix()
print(f"  关联矩阵: {assoc_matrix.shape}", flush=True)

company_lookup = load_company_lookup()
print(f"  公司画像: {len(company_lookup)} 家", flush=True)

history = load_stock_history()
print(f"  股价历史: {len(history)} 只股票", flush=True)

windows = build_windows()
print(f"  窗口数: {len(windows)}", flush=True)

# ---------- 步骤2: 推理样本构建 ----------
print()
print("=" * 60, flush=True)
print("Step 2: 构建 Task3 推理样本", flush=True)
print("=" * 60, flush=True)

from inference import build_inference_samples, load_artifacts

artifacts = load_artifacts()
category_map = artifacts["category_map"]
print(f"  类别映射: event_types={category_map['event_types']}", flush=True)
print(f"  类别映射: industry_types({len(category_map['industry_types'])})={category_map['industry_types'][:5]}...", flush=True)

all_predictions = []
selected_by_window = {}
all_ranked_by_window = {}

for window in windows:
    wid = window["window_id"]
    print(f"\n  --- 窗口{wid} ---", flush=True)

    # 筛选事件
    filtered_events = filter_events(events, window["publish_times"])
    print(f"  筛选后事件: {len(filtered_events)} 条", flush=True)

    if not filtered_events:
        print(f"  窗口{wid} 无有效事件，跳过", flush=True)
        continue

    # 构建公司池
    stock_pool = build_stock_pool(filtered_events, assoc_matrix, company_lookup)
    print(f"  候选股票: {len(stock_pool)} 只", flush=True)

    if not stock_pool:
        print(f"  窗口{wid} 无候选股票，跳过", flush=True)
        continue

    # 构建推理样本
    raw_samples = build_inference_samples(
        window_id=wid,
        events=filtered_events,
        stock_pool=stock_pool,
        company_lookup=company_lookup,
        history=history,
        category_map=category_map,
    )
    print(f"  推理样本: {len(raw_samples)} 条", flush=True)

    # ---------- 步骤3: Task3 推理 ----------
    print(f"  执行模型推理...", flush=True)

    from inference import load_model, run_inference

    model_artifacts = load_model()

    if model_artifacts is not None:
        predictions = run_inference(raw_samples, model_artifacts)
        print(f"  推理完成: {len(predictions)} 条", flush=True)
    else:
        # Mock 预测：基于 assoc_ij 模拟
        print(f"  [Mock] 模型未加载，使用模拟预测", flush=True)
        predictions = []
        for s in raw_samples:
            assoc = s["assoc_ij"]
            prob = 0.3 + 0.65 * (assoc - 0.6) / 0.6
            prob = max(0.05, min(0.95, prob))
            ret = prob * 0.05
            predictions.append({
                "window_id": wid,
                "event_id": s["event_id"],
                "publish_time": s["publish_time"],
                "event_type": s["event_type"],
                "stock_code": s["stock_code"],
                "stock_name": s["stock_name"],
                "assoc_ij": s["assoc_ij"],
                "pred_prob": round(prob, 4),
                "pred_return": round(ret, 6),
            })
        print(f"  Mock 预测: {len(predictions)} 条", flush=True)

    all_predictions.extend(predictions)

    # ---------- 步骤4: 融合 + 选股 ----------
    print(f"  执行融合评分...", flush=True)

    from fusion import fuse_and_rank

    selected, ranked = fuse_and_rank(predictions, top_k=3)
    selected_by_window[wid] = selected
    all_ranked_by_window[wid] = ranked
    print(f"  融合完成: {len(ranked)} 只候选 → 选中 {len(selected)} 只", flush=True)

    for r in selected:
        print(f"    rank={r['rank']} {r['stock_code']} {r['stock_name']}  "
              f"prob={r['fused_prob']:.4f} ret={r['fused_return']:.4f} "
              f"score={r['score']:.6f}", flush=True)

# ---------- 步骤5: 保存预测结果 ----------
predictions_path = OUTPUT_DIR / "task3_predictions.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
predictions_path.write_text(json.dumps(all_predictions, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n  预测结果已保存: {predictions_path}", flush=True)

# ---------- 步骤6: 交易模拟 ----------
print()
print("=" * 60, flush=True)
print("Step 3: 交易模拟", flush=True)
print("=" * 60, flush=True)

from trading import simulate_trading

summary = simulate_trading(
    selected_stocks_by_window=selected_by_window,
    windows=windows,
    initial_capital=100_000.0,
)

# ---------- 步骤7: 保存结果 ----------
print()
print("=" * 60, flush=True)
print("Step 4: 保存结果", flush=True)
print("=" * 60, flush=True)

# 窗口结果 CSV
window_rows = []
for wr in summary["window_results"]:
    for pos in wr["positions"]:
        window_rows.append({
            "window_id": wr["window_id"],
            "capital_before": wr["capital_before"],
            "profit": wr["profit"],
            "return_rate": wr["return_rate"],
            "capital_after": wr["capital_after"],
            "stock_code": pos["stock_code"],
            "stock_name": pos["stock_name"],
            "weight": pos["weight"],
            "buy_open": pos["buy_open"],
            "shares": pos["shares"],
            "cost": pos["cost"],
            "sell_close": pos["sell_close"],
            "revenue": pos["revenue"],
        })

window_df = pd.DataFrame(window_rows)
window_path = OUTPUT_DIR / "window_trade_results.csv"
window_df.to_csv(window_path, index=False, encoding="utf-8-sig")
print(f"  窗口交易结果: {window_path}", flush=True)

# 回测摘要 JSON
summary_path = OUTPUT_DIR / "backtest_summary.json"
summary_serializable = {k: v for k, v in summary.items() if k not in ["window_results", "capital_history"]}
summary_serializable["windows_detail"] = [
    {
        "window_id": wr["window_id"],
        "capital_before": wr["capital_before"],
        "profit": wr["profit"],
        "return_rate": wr["return_rate"],
        "capital_after": wr["capital_after"],
        "n_positions": wr["n_positions"],
    }
    for wr in summary["window_results"]
]
summary_serializable["final_summary"] = {
    "initial_capital": summary["initial_capital"],
    "final_capital": summary["final_capital"],
    "total_profit": summary["total_profit"],
    "total_return_rate": summary["total_return_rate"],
    "sharpe_ratio": summary["sharpe_ratio"],
    "max_drawdown": summary["max_drawdown"],
    "n_windows": summary["n_windows"],
    "n_profitable": summary["n_profitable"],
}
summary_path.write_text(json.dumps(summary_serializable, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  回测摘要: {summary_path}", flush=True)

# ---------- 最终打印 ----------
print()
print("=" * 60, flush=True)
print("回测完成", flush=True)
print("=" * 60, flush=True)
print(f"  初始资金: {summary['initial_capital']:,.2f}", flush=True)
print(f"  最终资金: {summary['final_capital']:,.2f}", flush=True)
print(f"  总收益:   {summary['total_profit']:,.2f}", flush=True)
print(f"  收益率:   {summary['total_return_rate']:.4%}", flush=True)
print(f"  夏普比率: {summary['sharpe_ratio']:.4f}", flush=True)
print(f"  最大回撤: {summary['max_drawdown']:.4%}", flush=True)
print(f"  盈利窗口: {summary['n_profitable']}/{summary['n_windows']}", flush=True)
print()
print(f"输出目录: {OUTPUT_DIR.resolve()}", flush=True)

if __name__ == "__main__":
    pass
