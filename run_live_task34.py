"""
实测 Task3+4：模型推理 → 融合选股 → 输出投资策略

核心规则：
- K=15：每个事件取语义关联强度最高的 15 家公司
- 模型输入：每个 (事件i, 公司j) 对
- 模型输出：pred_prob（上涨概率）+ pred_return（预测收益幅度）
- 融合：同一公司被多事件影响时，概率叠加，收益累加
- 选股：Score = fused_prob × fused_return，降序取 Top-3
- 资金分配：weight_i = score_i / Σscore_k
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# 把 task4 目录加入 sys.path，解决 inference.py 中的 from data_loader import ...
sys.path.insert(0, str(Path(__file__).parent / "task4"))

# ---------- 实测路径配置 ----------
PROJECT_DIR = Path(r"D:\Math_match\codes")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "live_task4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFICATION_PATH = PROJECT_DIR / "outputs" / "live_classification_eastmoney30" / "classification_results.json"
S0_TOPK_PATH = PROJECT_DIR / "outputs" / "live_task2_semantic" / "s0_topk.json"
COMPANY_PROFILES_PATH = PROJECT_DIR / "outputs" / "task2_profile_run" / "company_profiles.json"
MODEL_DIR = PROJECT_DIR / "outputs" / "task3" / "paper_training_lstm"
STOCK_HISTORY_PATHS = [
    PROJECT_DIR / "task3" / "hs300_history_batch1.json",
    PROJECT_DIR / "task3" / "hs300_history_batch2.json",
]

# ===== 核心参数 =====
K = 15               # 每个事件取关联强度最高的 K 家公司
TOP_K_SELECT = 3     # 融合后选股数量
S0_THRESHOLD = 0.7   # 语义关联分数最低阈值

# 实测窗口：4.20 事件 → 4.21 买入 → 4.23 卖出
WINDOWS = [
    {"window_id": 1, "publish_times": {"4.20"}, "buy_date": "20260421", "sell_date": "20260423"},
]

# 事件筛选阈值（实测放宽）
EVENT_FILTER = {
    "heat_min": 0.3,
    "intensity_min": 0.3,
    "range_min": 0.3,
    "attribute_min": 0.3,
}


# ---------- Step 1: 加载数据 ----------
print("=" * 60, flush=True)
print("Step 1: 加载实测数据", flush=True)
print("=" * 60, flush=True)

# 1a. 加载分类结果（事件）
classification_data = json.loads(CLASSIFICATION_PATH.read_text(encoding="utf-8"))
events = []
for item in classification_data:
    events.append({
        "sample_id": item["sample_id"],
        "publish_time": item.get("publish_time", ""),
        "event_type": item.get("event_type", ""),
        "duration_type": item.get("duration_type", ""),
        "heat": float(item.get("heat", 0)),
        "event_intensity": float(item.get("event_intensity", 0)),
        "influence_range": float(item.get("influence_range", 0)),
        "attribute_score": float(item.get("attribute_score", 0)),
        "event_summary": item.get("event_summary", ""),
        "title": item.get("title", ""),
    })
print(f"  事件总数: {len(events)}", flush=True)

# 1b. 加载公司画像
from task4.data_loader import normalize_stock_code

company_payload = json.loads(COMPANY_PROFILES_PATH.read_text(encoding="utf-8"))
company_items = company_payload.get("companies", company_payload) if isinstance(company_payload, dict) else company_payload
company_lookup = {}
for c in company_items:
    code = normalize_stock_code(c.get("stock_code", ""))
    if code:
        company_lookup[code] = c
print(f"  公司画像: {len(company_lookup)} 家", flush=True)

# 1c. 加载语义匹配 topk 结果
s0_topk_data = json.loads(S0_TOPK_PATH.read_text(encoding="utf-8"))
s0_topk_map = {row["sample_id"]: row for row in s0_topk_data}
print(f"  语义匹配: {len(s0_topk_data)} 条事件有 top-k 结果", flush=True)

# 1d. 加载股价历史
import task4.data_loader as dl_module
dl_module.STOCK_HISTORY_PATHS = STOCK_HISTORY_PATHS
from task4.data_loader import load_stock_history
history = load_stock_history()
print(f"  股价历史: {len(history)} 只股票", flush=True)


# ---------- Step 2: 事件筛选 + 公司池构建（K=15） ----------
print()
print("=" * 60, flush=True)
print(f"Step 2: 窗口事件筛选 + 公司池构建 (K={K})", flush=True)
print("=" * 60, flush=True)


def filter_events(events_list, publish_times):
    """按阈值筛选事件"""
    filtered = []
    for e in events_list:
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


def build_stock_pool_k15(filtered_events, s0_topk_map_local, company_lookup_local):
    """
    每个事件取语义关联强度最高的 K 家公司，合并去重
    返回 stock_pool 列表，每个元素含 stock_code / stock_name / assoc_ij
    """
    seen = set()
    pool = []
    event_company_pairs = []  # 记录每个事件关联了哪些公司

    for evt in filtered_events:
        sid = evt["sample_id"]
        topk_row = s0_topk_map_local.get(sid)
        if not topk_row:
            event_company_pairs.append((sid, []))
            continue

        companies_for_event = []
        for comp in topk_row.get("top_companies", [])[:K]:
            code = str(comp["stock_code"]).zfill(6)
            s0_score = comp.get("s0", 0)
            if s0_score < S0_THRESHOLD:
                continue
            if code not in company_lookup_local:
                continue
            companies_for_event.append({
                "stock_code": code,
                "stock_name": company_lookup_local[code].get("stock_name", code),
                "assoc_ij": s0_score,
                "match_priority": company_lookup_local[code].get("match_priority", "高"),
                "confidence": float(company_lookup_local[code].get("confidence", 0)),
                "industry_lv1": company_lookup_local[code].get("industry_lv1", ""),
            })
            if code not in seen:
                seen.add(code)
                pool.append(companies_for_event[-1])

        event_company_pairs.append((sid, companies_for_event))

    # 去重后的 pool 是所有事件涉及公司的并集
    # 但推理时每个事件只对自己的 K=15 公司预测
    return pool, event_company_pairs


# ---------- Step 3: 推理 + 融合 ----------
all_predictions = []
selected_by_window = {}

# 加载模型（只加载一次）
import task4.inference as inf_module
inf_module.MODEL_DIR = MODEL_DIR
inf_module.MODEL_PATH = MODEL_DIR / "best_model.pt"
inf_module.ARTIFACTS_PATH = MODEL_DIR / "artifacts.pkl"

from task4.inference import build_inference_samples, load_artifacts, load_model, run_inference

print("  加载 Task3 模型...", flush=True)
artifacts = load_artifacts()
category_map = artifacts["category_map"]
model_artifacts = load_model()

for window in WINDOWS:
    wid = window["window_id"]
    print(f"\n  --- 窗口{wid} ---", flush=True)

    # 筛选事件
    filtered = filter_events(events, window["publish_times"])
    print(f"  筛选后事件: {len(filtered)} 条", flush=True)

    if not filtered:
        print(f"  窗口{wid} 无有效事件，跳过")
        continue

    # 构建公司池：每个事件 K=15
    stock_pool, event_company_pairs = build_stock_pool_k15(filtered, s0_topk_map, company_lookup)
    print(f"  公司池（去重并集）: {len(stock_pool)} 只", flush=True)
    for sid, comps in event_company_pairs:
        print(f"    事件 {sid}: {len(comps)} 家公司", flush=True)

    # 按每个事件单独推理（每个事件只对自己的 K 家公司预测）
    for evt, (sid, comps) in zip(filtered, event_company_pairs):
        if not comps:
            continue

        raw_samples = build_inference_samples(
            window_id=wid,
            events=[evt],
            stock_pool=comps,
            company_lookup=company_lookup,
            history=history,
            category_map=category_map,
        )

        if model_artifacts is not None:
            predictions = run_inference(raw_samples, model_artifacts)
        else:
            # Mock
            predictions = []
            for s in raw_samples:
                assoc = s["assoc_ij"]
                prob = 0.3 + 0.65 * (assoc - 0.6) / 0.4
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

        all_predictions.extend(predictions)

    print(f"  总预测样本: {len(all_predictions)} 条", flush=True)

    # 融合评分 + 选股
    from task4.fusion import fuse_and_rank

    selected, ranked = fuse_and_rank(all_predictions, top_k=TOP_K_SELECT)
    selected_by_window[wid] = selected
    print(f"  融合完成: {len(ranked)} 只候选 → 选中 {len(selected)} 只", flush=True)

    for r in selected:
        print(f"    rank={r['rank']} {r['stock_code']} {r['stock_name']}  "
              f"上涨概率={r['fused_prob']:.2%}  预测收益={r['fused_return']:.4f}  "
              f"评分={r['score']:.6f}  资金占比={r['weight']:.2%}", flush=True)


# ---------- Step 4: 保存结果 ----------
print()
print("=" * 60, flush=True)
print("Step 3: 保存结果", flush=True)
print("=" * 60, flush=True)

# 保存全部预测（每个事件×公司的 pred_prob + pred_return）
predictions_path = OUTPUT_DIR / "task3_predictions.json"
predictions_path.write_text(json.dumps(all_predictions, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  全部预测: {predictions_path}", flush=True)

# 保存选股结果
all_selected = []
for wid, stocks in selected_by_window.items():
    all_selected.extend(stocks)

if all_selected:
    selected_df = pd.DataFrame(all_selected)
    selected_path = OUTPUT_DIR / "selected_stocks.csv"
    selected_df.to_csv(selected_path, index=False, encoding="utf-8-sig")
    print(f"  选股结果: {selected_path}", flush=True)


# ---------- 最终输出 ----------
print()
print("=" * 60, flush=True)
print("投资策略输出", flush=True)
print("=" * 60, flush=True)

for wid, stocks in selected_by_window.items():
    print(f"\n  窗口{wid} 选股结果 (Top-{TOP_K_SELECT}):", flush=True)
    for r in stocks:
        direction = "看涨" if r["fused_prob"] > 0.5 else "看跌"
        print(f"    {r['rank']}. {r['stock_code']} {r['stock_name']}  "
              f"[{direction}]  上涨概率={r['fused_prob']:.2%}  "
              f"预测收益={r['fused_return']:.4f}  "
              f"资金占比={r['weight']:.2%}", flush=True)

print(f"\n  模型说明:", flush=True)
print(f"    - 每个事件取语义关联 Top-{K} 公司", flush=True)
print(f"    - 模型输出: pred_prob(上涨概率) + pred_return(预测收益)", flush=True)
print(f"    - 融合规则: fused_prob = min(Σpred_prob, 1), fused_return = Σpred_return", flush=True)
print(f"    - 评分: Score = fused_prob × fused_return", flush=True)
print(f"    - 选股: Score 降序取 Top-{TOP_K_SELECT}", flush=True)
print(f"    - 资金分配: weight = score / Σscore", flush=True)
print(f"\n输出目录: {OUTPUT_DIR.resolve()}", flush=True)
