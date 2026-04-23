"""
Task 3 异常值检测模块

对所有训练/验证/待推理样本计算 AR 和 CAR(4)，
用 IQR / 标准差 方法识别异常值，输出可视化图表和处理结果。

公式：
  R_{i,t}     = (P_{i,t} - P_{i,t-1}) / P_{i,t-1}
  AR_{i,t}    = R_{i,t} - R_{m,t}
  CAR(4)_{ij} = Σ_{k=0}^{3} AR_{i,φ_k(j)}
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ---------- 路径 ----------
DATASET_DIR = Path(r"D:\Math_match\codes\task3\dataset")
PRICE_FILES = [
    Path(r"D:\Math_match\codes\task3\hs300_history_batch1.json"),
    Path(r"D:\Math_match\codes\task3\hs300_history_batch2.json"),
]
OUTPUT_DIR = Path(r"D:\Math_match\codes\outputs\task3\outlier_detection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- 1. 加载股价历史 ----------
def load_price_history():
    """返回 {stock_code: {date: row}}"""
    merged = defaultdict(dict)
    for p in PRICE_FILES:
        payload = json.loads(p.read_text(encoding="utf-8"))
        for stock_code, rows in payload.get("data_by_stock", {}).items():
            code = str(stock_code).zfill(6)
            for row in rows:
                d = str(row.get("日期", "")).strip()
                if not d:
                    continue
                merged[code][d] = {
                    "date": d,
                    "open":   float(row.get("开盘") or 0),
                    "close":  float(row.get("收盘") or 0),
                    "high":   float(row.get("最高") or 0),
                    "low":    float(row.get("最低") or 0),
                    "volume": float(row.get("成交量") or 0),
                    "amount": float(row.get("成交额") or 0),
                    "amplitude": float(row.get("振幅") or 0),
                    "pct_change": float(row.get("涨跌幅") or 0),
                    "change_amount": float(row.get("涨跌额") or 0),
                    "turnover_rate": float(row.get("换手率") or 0),
                }
    # 按日期排序
    for code in merged:
        merged[code] = dict(sorted(merged[code].items()))
    return merged


def get_close(history, stock_code, date):
    row = history.get(str(stock_code).zfill(6), {}).get(date)
    return row["close"] if row else None


# ---------- 2. 计算日收益率 ----------
def compute_daily_return(history, stock_code, date):
    """R_{i,t} = (P_t - P_{t-1}) / P_{t-1}"""
    code_history = history.get(str(stock_code).zfill(6), {})
    dates = sorted(code_history.keys())
    if date not in dates:
        return None
    idx = dates.index(date)
    if idx == 0:
        return None
    prev_date = dates[idx - 1]
    p_curr = code_history[date]["close"]
    p_prev = code_history[prev_date]["close"]
    if p_prev <= 0:
        return None
    return (p_curr - p_prev) / p_prev


# ---------- 3. 计算市场基准日收益率（等权法）----------
def compute_market_return(history, date):
    """R_{m,t} = 等权平均日收益率"""
    rets = []
    for code, by_date in history.items():
        if date not in by_date:
            continue
        dates = sorted(by_date.keys())
        idx = dates.index(date)
        if idx == 0:
            continue
        prev_date = dates[idx - 1]
        p_curr = by_date[date]["close"]
        p_prev = by_date[prev_date]["close"]
        if p_prev > 0:
            rets.append((p_curr - p_prev) / p_prev)
    return np.mean(rets) if rets else None


# ---------- 4. 计算单样本的每日 AR ----------
def compute_ar_for_sample(history, stock_code, date, mkt_return):
    """AR_{i,t} = R_{i,t} - R_{m,t}"""
    r = compute_daily_return(history, stock_code, date)
    if r is None or mkt_return is None:
        return None
    return r - mkt_return


# ---------- 5. 计算 CAR(4) ----------
def compute_car4(history, stock_code, event_date):
    """
    CAR(4) = Σ_{k=0}^{3} AR_{i,φ_k(j)}
    φ_k(j) = 事件日后的第 k 个交易日

    注意：标签 future_4day_return = (P_last - P_first) / P_first
    而非每日收益率累加。AR 体系下 CAR = Σ AR_k。
    """
    code_history = history.get(str(stock_code).zfill(6), {})
    dates = sorted(code_history.keys())

    # 找事件日后的交易日序列
    event_idx = None
    for i, d in enumerate(dates):
        if d >= event_date:
            event_idx = i
            break

    if event_idx is None:
        return None, []

    # 取事件日后 4 个交易日
    car_dates = dates[event_idx: event_idx + 4]
    if len(car_dates) < 4:
        car_dates = car_dates + [None] * (4 - len(car_dates))

    ar_list = []
    for d in car_dates:
        if d is None:
            ar_list.append(None)
        else:
            mkt = compute_market_return(history, d)
            ar = compute_ar_for_sample(history, stock_code, d, mkt)
            ar_list.append(ar)

    # AR-based CAR = 前 4 个有效 AR 的和
    valid_ars = [a for a in ar_list[:4] if a is not None]
    car = sum(valid_ars) if valid_ars else None
    return car, list(zip(car_dates, ar_list))


# ---------- 6. 加载样本集 ----------
def load_all_samples():
    datasets = {
        "train":   json.loads((DATASET_DIR / "train_samples_v1.json").read_text(encoding="utf-8"))["samples"],
        "val":     json.loads((DATASET_DIR / "val_samples_v1.json").read_text(encoding="utf-8"))["samples"],
        "pending": json.loads((DATASET_DIR / "pending_inference_samples_v1.json").read_text(encoding="utf-8"))["samples"],
        "test":    json.loads((DATASET_DIR / "test_samples_v2_custom_414.json").read_text(encoding="utf-8"))["samples"],
    }
    return datasets


# ---------- 7. 对所有样本计算 CAR ----------
def compute_all_cars(samples_dict, history):
    """
    返回 list of dict:
      {sample_id, split, event_id, event_date, stock_code, stock_name,
       car4, ar_days: [{date, ar, stock_return, mkt_return}], has_label}
    """
    results = []
    for split, samples in samples_dict.items():
        for s in samples:
            sid = s.get("sample_id", "")
            event_id = s.get("event_id", "")
            # 解析 event_date: 可能是 "20260414"（test）或 "20260411"（train/val）
            raw_date = str(s.get("event_date", ""))
            if raw_date.isdigit() and len(raw_date) == 8:
                event_date = raw_date  # 已经是 YYYYMMDD 格式
            else:
                # 处理 "4.11" 格式
                try:
                    parts = raw_date.strip().split(".")
                    m, d = int(parts[0]), int(parts[1])
                    event_date = f"2026{m:02d}{d:02d}"
                except Exception:
                    event_date = raw_date

            stock_code = str(s.get("stock_code", "")).zfill(6)
            stock_name = s.get("stock_name", stock_code)

            car4, ar_days = compute_car4(history, stock_code, event_date)

            # 同时获取原始 label（若存在）
            targets = s.get("targets", {}) or {}
            true_car = targets.get("future_4day_return")
            true_up = targets.get("future_4day_up")

            results.append({
                "sample_id": sid,
                "split": split,
                "event_id": event_id,
                "event_date": event_date,
                "stock_code": stock_code,
                "stock_name": stock_name,
                "car4": round(car4, 6) if car4 is not None else None,
                "n_valid_ar": sum(1 for a in ar_days[:4] if a[1] is not None),
                "ar_days": ar_days[:4],
                "true_car": true_car,
                "true_up": true_up,
                "has_label": car4 is not None,
            })
    return results


# ---------- 8. 异常值检测 ----------
def detect_outliers_iqr(car_values: list, multiplier: float = 1.5):
    """
    IQR 法：Q1 - multiplier*IQR 到 Q3 + multiplier*IQR 之外为异常值
    """
    vals = [v for v in car_values if v is not None]
    if len(vals) < 3:
        return []
    q1 = np.percentile(vals, 25)
    q3 = np.percentile(vals, 75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    outliers = [v for v in vals if v < lower or v > upper]
    return outliers, lower, upper, q1, q3


def detect_outliers_std(car_values: list, n_std: float = 2.0):
    """
    标准差法：均值 ± n_std * 标准差 之外为异常值
    """
    vals = [v for v in car_values if v is not None]
    if len(vals) < 3:
        return []
    mean = np.mean(vals)
    std = np.std(vals, ddof=1)
    lower = mean - n_std * std
    upper = mean + n_std * std
    outliers = [v for v in vals if v < lower or v > upper]
    return outliers, lower, upper, mean, std


# ---------- 9. 可视化 ----------
def plot_car_distribution(car_list, iqr_bounds, std_bounds, split_name, out_path):
    """CAR 分布直方图 + IQR/Std 异常值标记"""
    vals = [c["car4"] for c in car_list if c["car4"] is not None]
    if not vals:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    # 左图：直方图
    ax = axes[0]
    ax.hist(vals, bins=30, color="#5B7FA3", edgecolor="white", alpha=0.85)
    iqr_lo, iqr_hi = iqr_bounds[0], iqr_bounds[1]
    std_lo, std_hi = std_bounds[0], std_bounds[1]
    ymax = ax.get_ylim()[1]
    for b, color, label in [(iqr_lo, "red", f"IQR下界={iqr_lo:.4f}"),
                              (iqr_hi, "red", f"IQR上界={iqr_hi:.4f}"),
                              (std_lo, "orange", f"2σ下界={std_lo:.4f}"),
                              (std_hi, "orange", f"2σ上界={std_hi:.4f}")]:
        ax.axvline(b, color=color, linestyle="--", linewidth=1.5, label=label)
    ax.axvline(np.mean(vals), color="navy", linestyle="-", linewidth=2, label=f"均值={np.mean(vals):.4f}")
    ax.set_xlabel(r"$\hat{Y}_{ij}^{reg}$", fontsize=11)
    ax.set_ylabel("频数", fontsize=11)
    ax.set_title(f"{split_name} " + r"$\hat{Y}_{ij}^{reg}$" + f" 分布（n={len(vals)}）", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)

    # 右图：箱型图
    ax2 = axes[1]
    bp = ax2.boxplot(vals, vert=True, patch_artist=True,
                     boxprops=dict(facecolor="#5B7FA3", alpha=0.7),
                     medianprops=dict(color="navy", linewidth=2),
                     whiskerprops=dict(color="#5B7FA3"),
                     capprops=dict(color="#5B7FA3"),
                     flierprops=dict(marker="o", markerfacecolor="red", markersize=6))
    # 标注 IQR 边界
    ax2.axhline(iqr_lo, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax2.axhline(iqr_hi, color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"IQR [{iqr_lo:.4f}, {iqr_hi:.4f}]")
    ax2.set_ylabel(r"$\hat{Y}_{ij}^{reg}$", fontsize=11)
    ax2.set_title(f"{split_name} " + r"$\hat{Y}_{ij}^{reg}$" + " 箱型图", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Plot] {out_path}")


def plot_scatter_outliers(results, out_path, name="全部样本"):
    """散点图：每个样本的 CAR4，异常值标红"""
    has_car = [r for r in results if r["car4"] is not None]
    if not has_car:
        return

    vals = [r["car4"] for r in has_car]
    iqr_out, iqr_lo, iqr_hi, _, _ = detect_outliers_iqr(vals)
    std_out, std_lo, std_hi, _, _ = detect_outliers_std(vals)
    iqr_set = set(iqr_out)
    std_set = set(std_out)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f8f8")

    x = range(len(has_car))
    colors = []
    for r, v in zip(has_car, vals):
        if v in iqr_set:
            colors.append("red")
        elif v in std_set:
            colors.append("orange")
        else:
            colors.append("#5B7FA3")

    ax.scatter(x, vals, c=colors, s=40, alpha=0.8, zorder=3)
    ax.axhline(0, color="gray", linewidth=1, zorder=2)
    ax.axhline(iqr_lo, color="red", linestyle="--", linewidth=1, label=f"IQR [{iqr_lo:.4f}, {iqr_hi:.4f}]")
    ax.axhline(iqr_hi, color="red", linestyle="--", linewidth=1)
    ax.axhline(np.mean(vals), color="navy", linewidth=1.5, label=f"均值={np.mean(vals):.4f}")

    ax.set_xlabel("样本索引", fontsize=11)
    ax.set_ylabel(r"$\hat{Y}_{ij}^{reg}$", fontsize=11)
    ax.set_title(f"{name} " + r"$\hat{Y}_{ij}^{reg}$" + " 散点图（红=IQR异常 橙=仅Std异常 蓝=正常）", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Plot] {out_path}")


# ---------- 主函数 ----------
def main() -> int:
    print("=" * 60)
    print("Task 3 异常值检测")
    print("=" * 60)

    # 1. 加载数据
    history = load_price_history()
    print(f"[Data] 股价历史: {len(history)} 只股票")

    samples_dict = load_all_samples()
    for k, v in samples_dict.items():
        print(f"[Data] {k}: {len(v)} 样本")

    # 2. 计算所有 CAR
    results = compute_all_cars(samples_dict, history)
    print(f"[CAR] 共计算 {len(results)} 条样本的 CAR(4)")

    # 有标签的
    labeled = [r for r in results if r["has_label"]]
    unlabeled = [r for r in results if not r["has_label"]]
    print(f"[CAR] 有CAR: {len(labeled)}  条  无CAR: {len(unlabeled)} 条")

    # 3. 异常值检测
    all_cars = [r["car4"] for r in labeled]

    iqr_out, iqr_lo, iqr_hi, q1, q3 = detect_outliers_iqr(all_cars)
    std_out, std_lo, std_hi, mean, std = detect_outliers_std(all_cars)

    print(f"\n[异常值检测]")
    print(f"  IQR 法: 边界 [{iqr_lo:.4f}, {iqr_hi:.4f}], 异常值 {len(iqr_out)} 个")
    print(f"  Std 法: 边界 [{std_lo:.4f}, {std_hi:.4f}], 异常值 {len(std_out)} 个")
    print(f"  均值={mean:.4f}  标准差={std:.4f}  Q1={q1:.4f}  Q3={q3:.4f}")

    # 4. 标记每个样本
    iqr_set = set(iqr_out)
    std_set = set(std_out)

    for r in labeled:
        v = r["car4"]
        r["outlier_iqr"] = v in iqr_set
        r["outlier_std"] = v in std_set
        r["outlier_level"] = "IQR" if v in iqr_set else ("Std" if v in std_set else "Normal")

    # 5. 输出异常值列表
    outlier_rows = [r for r in labeled if r["outlier_iqr"] or r["outlier_std"]]
    outlier_rows.sort(key=lambda x: x["car4"], reverse=True)

    print(f"\n[异常值样本] 共 {len(outlier_rows)} 条:")
    for r in outlier_rows[:20]:
        print(f"  [{r['outlier_level']}] {r['sample_id']}  {r['stock_code']} {r['stock_name']}  CAR={r['car4']:.4f}  事件={r['event_id']}")

    # 6. 保存 CSV
    df = pd.DataFrame(labeled)
    csv_path = OUTPUT_DIR / "car4_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[Output] CAR结果: {csv_path}")

    outlier_df = pd.DataFrame(outlier_rows)
    outlier_path = OUTPUT_DIR / "outliers.csv"
    outlier_df.to_csv(outlier_path, index=False, encoding="utf-8-sig")
    print(f"[Output] 异常值: {outlier_path}")

    # 7. 可视化
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 按 split 分组绘图
    for split in ["train", "val", "test"]:
        split_data = [r for r in labeled if r["split"] == split]
        if split_data:
            split_cars = [r["car4"] for r in split_data]
            siqr = detect_outliers_iqr(split_cars)
            sstd = detect_outliers_std(split_cars)
            out = OUTPUT_DIR / f"car4_dist_{split}.png"
            plot_car_distribution(split_data, (siqr[1], siqr[2]), (sstd[1], sstd[2]), split, out)

    # 全量散点图
    scatter_path = OUTPUT_DIR / "car4_scatter_all.png"
    plot_scatter_outliers(labeled, scatter_path, name="全部样本")

    # 测试集散点图（仅 test）
    test_data = [r for r in labeled if r["split"] == "test"]
    if test_data:
        scatter_test_path = OUTPUT_DIR / "car4_scatter_test.png"
        plot_scatter_outliers(test_data, scatter_test_path, name="test")

    # 8. 统计摘要（包含分split统计）
    per_split_stats = {}
    for split in ["train", "val", "test"]:
        split_data = [r for r in labeled if r["split"] == split]
        if not split_data:
            continue
        split_cars = [r["car4"] for r in split_data]
        siqr = detect_outliers_iqr(split_cars)
        sstd = detect_outliers_std(split_cars)
        per_split_stats[split] = {
            "count": len(split_data),
            "mean_car4": round(float(np.mean(split_cars)), 6),
            "std_car4": round(float(np.std(split_cars, ddof=1)), 6),
            "q1": round(siqr[3], 6),
            "q3": round(siqr[4], 6),
            "iqr_bounds": [round(siqr[1], 6), round(siqr[2], 6)],
            "std_bounds": [round(sstd[1], 6), round(sstd[2], 6)],
            "iqr_outlier_count": len(siqr[0]),
            "std_outlier_count": len(sstd[0]),
        }

    summary = {
        "total_labeled": len(labeled),
        "total_unlabeled": len(unlabeled),
        "iqr_bounds": [round(iqr_lo, 6), round(iqr_hi, 6)],
        "std_bounds": [round(std_lo, 6), round(std_hi, 6)],
        "iqr_outlier_count": len(iqr_out),
        "std_outlier_count": len(std_out),
        "mean_car4": round(mean, 6),
        "std_car4": round(std, 6),
        "q1": round(q1, 6),
        "q3": round(q3, 6),
        "per_split": per_split_stats,
        "outlier_samples": [
            {k: v for k, v in r.items() if k != "ar_days"}
            for r in outlier_rows
        ],
    }

    summary_path = OUTPUT_DIR / "outlier_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Output] 摘要: {summary_path}")

    print(f"\n输出目录: {OUTPUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
