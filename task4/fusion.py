"""
Task 4 Step 3: 预测融合与选股
- 概率融合：P_i^{(fus)} = min(Σ pred_prob, 1)
- 收益幅度融合：G_i^{(fus)} = Σ pred_return
- 综合评分：Score_i = P_i^{(fus)} × G_i^{(fus)}
- 选股：Score 降序，取 Top-3
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ALPHA = 0.7   # LSTM 权重（方案模板）
BETA = 0.3    # 情绪权重（预留，暂无舆情数据时设为0）
TOP_K = 3     # 每窗口选股数量

SENTIMENT_PATH = None  # 暂不支持舆情数据时为 None


def fuse_and_rank(predictions: list, top_k: int = TOP_K, sentiment_weight: float = BETA) -> list:
    """
    对同一窗口内同一公司的多个事件预测结果进行融合与排序

    Args:
        predictions: run_inference 的输出列表
        top_k: 每窗口选股数量
        sentiment_weight: 情绪权重 β（暂无舆情时用 0）

    Returns:
        list of dict: 含 window_id / stock_code / stock_name / fused_prob / fused_return / score / rank
    """
    # 按 (window_id, stock_code) 分组
    by_window_stock = defaultdict(list)
    for p in predictions:
        by_window_stock[(p["window_id"], p["stock_code"])].append(p)

    results = []
    for (wid, code), items in by_window_stock.items():
        # 概率融合：min(sum, 1)
        fused_prob = min(sum(p["pred_prob"] for p in items), 1.0)

        # 收益幅度融合
        fused_return = sum(p["pred_return"] for p in items)

        # 综合评分（暂不考虑情绪）
        score = fused_prob * fused_return

        results.append({
            "window_id": wid,
            "stock_code": code,
            "stock_name": items[0]["stock_name"],
            "event_count": len(items),
            "fused_prob": round(fused_prob, 4),
            "fused_return": round(fused_return, 6),
            "score": round(score, 6),
            "top_event": items[0]["event_id"],
        })

    # 按 score 降序
    results.sort(key=lambda x: x["score"], reverse=True)

    # 取 Top-K
    selected = results[:top_k]

    # 分配权重
    total_score = sum(r["score"] for r in selected if r["score"] > 0)
    if total_score > 0:
        for r in selected:
            r["weight"] = round(r["score"] / total_score, 4)
    else:
        for r in selected:
            r["weight"] = 0.0

    for rank, r in enumerate(selected, 1):
        r["rank"] = rank

    return selected, results  # selected=Top-K, results=全部排序


def run_fusion(predictions_path: Path, output_dir: Path, top_k: int = TOP_K):
    """
    读取预测结果，执行融合，保存选股结果
    """
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    print(f"[Fusion] 加载预测结果: {len(predictions)} 条", flush=True)

    # 按窗口分组输出
    windows = set(p["window_id"] for p in predictions)
    all_selected = []
    all_ranked = {}

    for wid in sorted(windows):
        window_preds = [p for p in predictions if p["window_id"] == wid]
        selected, ranked = fuse_and_rank(window_preds, top_k=top_k)
        all_selected.extend(selected)
        all_ranked[wid] = ranked

        print(f"[Fusion] 窗口{wid}: {len(ranked)} 只候选 → 选入 {len(selected)} 只", flush=True)
        for r in selected:
            print(f"  rank={r['rank']} {r['stock_code']} {r['stock_name']}  "
                  f"prob={r['fused_prob']:.4f}  ret={r['fused_return']:.4f}  "
                  f"score={r['score']:.6f}  weight={r['weight']:.4f}", flush=True)

    # 保存全部排序
    all_ranked_path = output_dir / "all_ranked_stocks.json"
    all_ranked_serializable = {str(k): v for k, v in all_ranked.items()}
    all_ranked_path.write_text(json.dumps(all_ranked_serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Fusion] 全量排序已保存: {all_ranked_path}", flush=True)

    # 保存选中股票
    selected_df = pd.DataFrame(all_selected)
    selected_path = output_dir / "selected_stocks.csv"
    selected_df.to_csv(selected_path, index=False, encoding="utf-8-sig")
    print(f"[Fusion] 选中股票已保存: {selected_path}", flush=True)

    return all_selected, all_ranked


if __name__ == "__main__":
    from pathlib import Path

    OUTPUT_DIR = Path(r"D:\Math_match\codes\outputs\task4")
    PREDICTIONS_PATH = OUTPUT_DIR / "task3_predictions.json"

    if PREDICTIONS_PATH.exists():
        selected, ranked = run_fusion(PREDICTIONS_PATH, OUTPUT_DIR)
        print(f"\n[Fusion] 完成，共选中 {len(selected)} 只股票")
    else:
        print(f"[Fusion] 预测文件不存在: {PREDICTIONS_PATH}")
