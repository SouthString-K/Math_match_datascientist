"""
实测 Task3+4（真实股价版）：基于4.17-4.20实际股价数据

规则：
- K=15：每个事件取语义关联强度最高的 15 家公司
- 使用真实股价数据构建 time_seq/delta_feat（与训练一致的10维）
- 模型输出：pred_prob（上涨概率）+ pred_return（预测收益幅度）
- 融合：同一公司被多事件影响时，概率叠加截断，收益累加
- 选股：Score = fused_prob × fused_return，降序取 Top-3
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent / "task4"))

# ---------- 路径配置 ----------
PROJECT_DIR = Path(r"D:\Math_match\codes")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "live_task4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFICATION_PATH = PROJECT_DIR / "outputs" / "live_classification_eastmoney30" / "classification_results.json"
S0_TOPK_PATH = PROJECT_DIR / "outputs" / "live_task2_semantic" / "s0_topk.json"
REAL_STOCK_PATH = OUTPUT_DIR / "hs300_20260417_20260420.json"
COMPANY_PROFILES_PATH = PROJECT_DIR / "outputs" / "task2_profile_run" / "company_profiles.json"
MODEL_DIR = PROJECT_DIR / "outputs" / "task3" / "paper_training_lstm"

# ===== 核心参数 =====
K = 15               # 每个事件关联 Top-K 公司
TOP_K_SELECT = 3     # 融合后选股数量
S0_THRESHOLD = 0.70  # 语义关联分数最低阈值

# ---------- Step 1: 加载数据 ----------
print("=" * 60)
print("Step 1: 加载数据")
print("=" * 60)

# 分类结果
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
print(f"  事件总数: {len(events)}")

# 语义匹配 topk
s0_topk_data = json.loads(S0_TOPK_PATH.read_text(encoding="utf-8"))
s0_topk_map = {row["sample_id"]: row for row in s0_topk_data}
print(f"  语义匹配: {len(s0_topk_data)} 条")

# 公司画像
from task4.data_loader import normalize_stock_code, safe_float
company_payload = json.loads(COMPANY_PROFILES_PATH.read_text(encoding="utf-8"))
company_items = company_payload.get("companies", company_payload) if isinstance(company_payload, dict) else company_payload
company_lookup = {}
for c in company_items:
    code = normalize_stock_code(c.get("stock_code", ""))
    if code:
        company_lookup[code] = c
print(f"  公司画像: {len(company_lookup)} 家")

# 真实股价数据 → 转为 history 格式 {code: [row_dicts]}
real_stock_raw = json.loads(REAL_STOCK_PATH.read_text(encoding="utf-8"))
real_stock_df = pd.DataFrame(real_stock_raw["data"])
print(f"  真实股价: {len(real_stock_df)} 条记录, {real_stock_df['股票代码'].nunique()} 只股票")

# 构建兼容 history 格式
history = defaultdict(list)
for _, row in real_stock_df.iterrows():
    code = str(row["股票代码"]).zfill(6)
    history[code].append({
        "date": str(row["日期"]),
        "stock_code": code,
        "open": safe_float(row.get("开盘")),
        "close": safe_float(row.get("收盘")),
        "high": safe_float(row.get("最高")),
        "low": safe_float(row.get("最低")),
        "volume": safe_float(row.get("成交量")),
        "amount": safe_float(row.get("成交额")),
        "amplitude": safe_float(row.get("振幅")),
        "pct_change": safe_float(row.get("涨跌幅")),
        "change_amount": safe_float(row.get("涨跌额")),
        "turnover_rate": safe_float(row.get("换手率")),
    })
# 排序
for code in history:
    history[code].sort(key=lambda r: r["date"])
history = dict(history)

# 构建原始 map 用于后续验证
stock_price_map = {}
for _, row in real_stock_df.iterrows():
    code = str(row["股票代码"]).zfill(6)
    date = str(row["日期"])
    if code not in stock_price_map:
        stock_price_map[code] = {}
    stock_price_map[code][date] = row.to_dict()

# ---------- Step 2: 构建事件-公司对（K=15） ----------
print()
print("=" * 60)
print(f"Step 2: 构建事件-公司对 (K={K}, S0≥{S0_THRESHOLD})")
print("=" * 60)

valid_events = [e for e in events if e["attribute_score"] >= 0.3 and e["sample_id"] in s0_topk_map]
print(f"  有效事件: {len(valid_events)} 条")

# 构建 stock_pool per event
event_stock_pools = {}  # {sample_id: [stock_info_list]}
all_involved_codes = set()
for evt in valid_events:
    sid = evt["sample_id"]
    topk_row = s0_topk_map[sid]
    pool = []
    for comp in topk_row.get("top_companies", []):
        if len(pool) >= K:
            break
        code = str(comp["stock_code"]).zfill(6)
        s0_score = comp.get("s0", 0)
        if s0_score < S0_THRESHOLD:
            continue
        if code not in history:
            continue
        pool.append({
            "stock_code": code,
            "stock_name": comp.get("stock_name", code),
            "assoc_ij": s0_score,
        })
        all_involved_codes.add(code)
    event_stock_pools[sid] = pool

total_pairs = sum(len(v) for v in event_stock_pools.values())
print(f"  事件-公司对: {total_pairs} 个")
print(f"  涉及公司: {len(all_involved_codes)} 只")

# ---------- Step 3: 使用标准 build_inference_samples + 模型推理 ----------
print()
print("=" * 60)
print("Step 3: LSTM模型推理（标准特征构建 + 真实股价）")
print("=" * 60)

from task4.inference import (
    load_artifacts, load_model, build_inference_samples, run_inference
)

artifacts = load_artifacts()
category_map = artifacts["category_map"]
model_artifacts = load_model()

if model_artifacts is not None:
    # 用标准函数构建样本（使用真实股价 history）
    all_samples = []
    for evt in valid_events:
        sid = evt["sample_id"]
        pool = event_stock_pools.get(sid, [])
        if not pool:
            continue
        samples = build_inference_samples(
            window_id=1,
            events=[evt],
            stock_pool=pool,
            company_lookup=company_lookup,
            history=history,
            category_map=category_map,
        )
        all_samples.extend(samples)
    
    print(f"  推理样本数: {len(all_samples)}")
    
    # 模型推理
    predictions = run_inference(all_samples, model_artifacts)
    print(f"  推理完成: {len(predictions)} 条预测")
else:
    print("  [警告] 模型加载失败，使用 mock 模式")
    predictions = []
    for evt in valid_events:
        sid = evt["sample_id"]
        for comp in event_stock_pools.get(sid, []):
            prob = 0.3 + 0.6 * (comp["assoc_ij"] - 0.7) / 0.3
            prob = max(0.1, min(0.9, prob))
            predictions.append({
                "window_id": 1,
                "event_id": sid,
                "event_type": evt["event_type"],
                "stock_code": comp["stock_code"],
                "stock_name": comp["stock_name"],
                "assoc_ij": comp["assoc_ij"],
                "pred_prob": round(prob, 4),
                "pred_return": round(prob * 0.04, 6),
            })

# ---------- Step 4: 融合评分 + 选股 ----------
print()
print("=" * 60)
print("Step 4: 融合评分 + 选股 (Top-3)")
print("=" * 60)

# 按股票分组融合
by_stock = defaultdict(list)
for p in predictions:
    by_stock[p["stock_code"]].append(p)

ranked = []
for code, items in by_stock.items():
    fused_prob = min(sum(p["pred_prob"] for p in items), 1.0)
    fused_return = sum(p["pred_return"] for p in items)
    score = fused_prob * fused_return
    
    # 查真实4.20涨跌幅（作为验证参考）
    real_420 = stock_price_map.get(code, {}).get("20260420", {})
    real_return_420 = float(real_420.get("涨跌幅", 0) or 0)
    
    ranked.append({
        "stock_code": code,
        "stock_name": items[0]["stock_name"],
        "event_count": len(items),
        "fused_prob": round(fused_prob, 4),
        "fused_return": round(fused_return, 6),
        "score": round(score, 6),
        "real_return_420": real_return_420,
        "top_events": [it["event_id"] for it in items[:3]],
    })

# 降序排序
ranked.sort(key=lambda x: x["score"], reverse=True)

# Top-3 选股
selected = ranked[:TOP_K_SELECT]

# 资金分配
total_score = sum(r["score"] for r in selected if r["score"] > 0)
if total_score > 0:
    for r in selected:
        r["weight"] = round(r["score"] / total_score, 4)
else:
    for r in selected:
        r["weight"] = round(1.0 / len(selected), 4)

for rank_i, r in enumerate(selected, 1):
    r["rank"] = rank_i

print(f"  候选公司: {len(ranked)} 只")
print(f"  选中 Top-{TOP_K_SELECT}:")
for r in selected:
    direction = "看涨" if r["fused_prob"] > 0.5 else "看跌"
    print(f"    {r['rank']}. {r['stock_code']} {r['stock_name']}  "
          f"[{direction}]  P={r['fused_prob']:.2%}  R={r['fused_return']:.4f}  "
          f"Score={r['score']:.6f}  Weight={r['weight']:.2%}  "
          f"(4.20实际涨跌={r['real_return_420']:+.2f}%)")

# ---------- Step 5: 保存结果 ----------
print()
print("=" * 60)
print("Step 5: 保存结果")
print("=" * 60)

# 保存预测
pred_path = OUTPUT_DIR / "real_predictions.json"
pred_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  全部预测: {pred_path}")

# 保存选股JSON
result = {
    "window_id": 1,
    "event_date": "20260420",
    "K": K,
    "S0_threshold": S0_THRESHOLD,
    "total_pairs": total_pairs,
    "total_candidates": len(ranked),
    "selected_stocks": selected,
    "all_ranked_top10": ranked[:10],
}
result_path = OUTPUT_DIR / "real_selected_stocks.json"
result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  选股结果: {result_path}")

# 打印完整 Top-10
print()
print("=" * 60)
print("完整排名 Top-10:")
print("=" * 60)
print(f"{'排名':<4} {'代码':<8} {'名称':<10} {'事件数':<6} {'融合概率':<8} {'预测收益':<10} {'评分':<10} {'4.20实际':<8}")
print("-" * 74)
for i, r in enumerate(ranked[:10], 1):
    print(f"{i:<4} {r['stock_code']:<8} {r['stock_name']:<10} {r['event_count']:<6} "
          f"{r['fused_prob']:.2%}  {r['fused_return']:+.4f}   {r['score']:.6f}  {r['real_return_420']:+.2f}%")

# ---------- Step 6: 模型准确性验证（预测 vs 真实） ----------
print()
print("=" * 60)
print("Step 6: 模型准确性验证")
print("=" * 60)

# 对每个预测样本，找到真实4.20涨跌数据
validation_records = []
for p in predictions:
    code = p["stock_code"]
    real_420 = stock_price_map.get(code, {}).get("20260420", {})
    real_pct = float(real_420.get("涨跌幅", 0) or 0)
    real_up = 1 if real_pct > 0 else 0  # 真实是否上涨
    pred_up = 1 if p["pred_prob"] > 0.5 else 0  # 预测是否上涨
    
    validation_records.append({
        "event_id": p["event_id"],
        "stock_code": code,
        "stock_name": p["stock_name"],
        "pred_prob": p["pred_prob"],
        "pred_return": p["pred_return"],
        "pred_direction": pred_up,
        "real_return_pct": real_pct,
        "real_direction": real_up,
        "direction_correct": int(pred_up == real_up),
        "return_error": abs(p["pred_return"] - real_pct / 100.0),
    })

# 计算指标
total = len(validation_records)
correct_dir = sum(r["direction_correct"] for r in validation_records)
acc_direction = correct_dir / total if total > 0 else 0

mae = np.mean([r["return_error"] for r in validation_records])
rmse = np.sqrt(np.mean([r["return_error"]**2 for r in validation_records]))

# 按公司去重后的验证（融合后）
fused_correct = sum(1 for r in ranked if
    (r["fused_prob"] > 0.5 and r["real_return_420"] > 0) or
    (r["fused_prob"] <= 0.5 and r["real_return_420"] <= 0)
)
fused_total = len(ranked)
fused_acc = fused_correct / fused_total if fused_total > 0 else 0

# Top-3 选股准确性
top3_correct = sum(1 for r in selected if r["real_return_420"] > 0)
top3_avg_real = np.mean([r["real_return_420"] for r in selected]) if selected else 0

print(f"  === 样本级验证（{total} 个事件-公司对） ===")
print(f"  涨跌方向准确率: {acc_direction:.2%} ({correct_dir}/{total})")
print(f"  收益幅度 MAE:    {mae:.4f}")
print(f"  收益幅度 RMSE:   {rmse:.4f}")
print()
print(f"  === 融合后验证（{fused_total} 只候选股票） ===")
print(f"  融合方向准确率: {fused_acc:.2%} ({fused_correct}/{fused_total})")
print()
print(f"  === Top-3 选股验证 ===")
print(f"  实际上涨命中: {top3_correct}/3")
print(f"  实际平均涨幅: {top3_avg_real:+.2f}%")
for r in selected:
    hit = "✓" if r["real_return_420"] > 0 else "✗"
    print(f"    {hit} {r['stock_code']} {r['stock_name']}  "
          f"预测={r['fused_return']:+.4f}  实际={r['real_return_420']:+.2f}%")

# 保存验证结果
validation_summary = {
    "sample_level": {
        "total_pairs": total,
        "direction_accuracy": round(acc_direction, 4),
        "direction_correct": correct_dir,
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
    },
    "fused_level": {
        "total_candidates": fused_total,
        "direction_accuracy": round(fused_acc, 4),
        "direction_correct": fused_correct,
    },
    "top3_selection": {
        "hit_count": top3_correct,
        "avg_real_return_pct": round(top3_avg_real, 4),
        "stocks": [{
            "stock_code": r["stock_code"],
            "stock_name": r["stock_name"],
            "pred_return": r["fused_return"],
            "real_return_pct": r["real_return_420"],
            "hit": r["real_return_420"] > 0,
        } for r in selected],
    },
}
val_path = OUTPUT_DIR / "validation_results.json"
val_path.write_text(json.dumps(validation_summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n  验证结果已保存: {val_path}")

print(f"\n完成！输出目录: {OUTPUT_DIR}")
