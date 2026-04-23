"""
绘制 K=15 验证集散点图（概率图 + 幅度图）

规则：
- 每个事件取 relation_score 最高的 K=15 家公司
- 模型输出：pred_prob（上涨概率 ŷ_cls）+ pred_return（预测收益 ŷ_reg）
- 对比真实标签：Y_cls（是否上涨）+ Y_reg（实际收益）
"""
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

# ========== 路径 ==========
MODEL_DIR  = Path(r"D:\Math_match\codes\outputs\task3\paper_training_lstm")
MODEL_PATH = MODEL_DIR / "best_model.pt"
ARTIFACTS  = MODEL_DIR / "artifacts.pkl"
ENRICHED   = MODEL_DIR / "enriched_dataset.json"

# 验证集（4.13 事件，作为测试集使用）
VAL_PATH   = Path(r"D:\Math_match\codes\task3\dataset\val_samples_v1.json")

OUTPUT_DIR = Path(r"D:\Math_match\codes\outputs\task3\paper_training_lstm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K = 15  # 每个事件取关联强度最高的 K 家公司

# ========== TF-IDF ==========
class CharTfidfVectorizer:
    def __init__(self, max_features=2000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary_ = {}
        self.idf_ = None

    @classmethod
    def from_vocabulary(cls, vocabulary, ngram_range=(1, 2)):
        obj = cls(max_features=len(vocabulary), ngram_range=ngram_range)
        obj.vocabulary_ = dict(vocabulary)
        obj.idf_ = np.ones(len(vocabulary), dtype=np.float32)
        return obj

    def fit(self, texts):
        df = {}
        for t in texts:
            for tok in set(self._tokenize(t)):
                df[tok] = df.get(tok, 0) + 1
        sorted_toks = sorted(df.items(), key=lambda x: (-x[1], x[0]))[:self.max_features]
        self.vocabulary_ = {t: i for i, (t, _) in enumerate(sorted_toks)}
        n = max(len(texts), 1)
        self.idf_ = np.ones(len(self.vocabulary_), dtype=np.float32)
        for t, i in self.vocabulary_.items():
            self.idf_[i] = math.log((1 + n) / (1 + df.get(t, 1))) + 1.0
        return self

    def _tokenize(self, text):
        tokens = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            if len(text) < n:
                continue
            for i in range(len(text) - n + 1):
                tokens.append(text[i:i+n])
        return tokens

    def transform(self, texts):
        n = len(self.vocabulary_)
        mat = np.zeros((len(texts), n), dtype=np.float32)
        for ri, text in enumerate(texts):
            tf = {}
            for t in self._tokenize(text):
                if t in self.vocabulary_:
                    tf[self.vocabulary_[t]] = tf.get(self.vocabulary_[t], 0) + 1
            for i, c in tf.items():
                mat[ri, i] = float(c) * self.idf_[i]
            norm = np.linalg.norm(mat[ri])
            if norm > 0:
                mat[ri] /= norm
        return mat


# ========== 加载模型 ==========
head = json.loads((MODEL_DIR / "head_params.json").read_text(encoding="utf-8"))
W_c, b_c = np.array(head["W_c"]), np.array(head["b_c"])

with open(ARTIFACTS, "rb") as f:
    art = pickle.load(f)
scalers = art["scalers"]
cat_map = art["category_map"]

# 使用训练时保存的词汇表
ev_vocab = CharTfidfVectorizer.from_vocabulary(art["event_vocabulary"], ngram_range=(1, 2))
co_vocab = CharTfidfVectorizer.from_vocabulary(art["company_vocabulary"], ngram_range=(1, 2))

import importlib.util
spec = importlib.util.spec_from_file_location("_m", Path(r"D:\Math_match\codes\task3\model.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PaperEventLSTM = mod.PaperEventLSTM

n_ind = len(cat_map["industry_types"])
n_chn = len(cat_map["chain_positions"])
net = PaperEventLSTM(
    event_text_dim=128,
    event_num_dim=4 + len(cat_map["event_types"]) + len(cat_map["duration_types"]),
    company_text_dim=192,
    company_num_dim=n_ind + n_chn + 1,
    time_input_dim=10, delta_dim=10,
    text_hidden_dim=64, num_hidden_dim=32,
    time_hidden_dim=32, fusion_hidden_dim=64, dropout=0.2,
)
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
net.load_state_dict(ckpt["model_state"], strict=False)
net.eval()

# ========== 加载公司画像 ==========
COMPANY_PROFILES_PATH = Path(r"D:\Math_match\codes\outputs\task2_profile_run\company_profiles.json")
company_payload = json.loads(COMPANY_PROFILES_PATH.read_text(encoding="utf-8"))
company_items = company_payload.get("companies", company_payload) if isinstance(company_payload, dict) else company_payload
company_lk = {}
for c in company_items:
    code = str(c.get("stock_code", "")).strip().lstrip("0").zfill(6)
    if code:
        company_lk[code] = c


# ========== 辅助函数 ==========
def safe_float(v, default=0.0):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except:
        return default

def one_hot(v, cats):
    return [1.0 if v == c else 0.0 for c in cats]

def multi_hot(vs, cats):
    s = set(vs or [])
    return [1.0 if c in s else 0.0 for c in cats]

PRICE_FIELDS = ["open", "close", "high", "low", "volume", "amount",
                "amplitude", "pct_change", "change_amount", "turnover_rate"]

def extract_price(row):
    if row is None:
        return [0.0] * len(PRICE_FIELDS)
    return [safe_float(row.get(f, 0)) for f in PRICE_FIELDS]

def compute_delta(rows):
    if len(rows) < 2:
        return [0.0] * len(PRICE_FIELDS)
    a = np.array(extract_price(rows[0]), dtype=np.float32)
    b = np.array(extract_price(rows[1]), dtype=np.float32)
    return (b - a).tolist()

def apply_std(mat, scaler):
    mean = np.array(scaler["mean"], dtype=np.float32)
    std  = np.array(scaler["std"],  dtype=np.float32)
    std[std < 1e-6] = 1.0
    return ((mat - mean) / std).astype(np.float32)


def build_sample(s):
    """从 val/test 数据集样本构建模型输入"""
    ev_text = s.get("event_features", {}).get("event_text", "") or ""
    if not ev_text:
        ev_text = f"{s.get('event_title', '')} {s.get('event_summary', '')}".strip()
    code = str(s.get("stock_code", "")).zfill(6)
    company = company_lk.get(code, {})
    co_parts = [company.get("stock_name", ""),
                company.get("summary_for_matching", ""),
                company.get("business_desc", "")]
    co_text = " ".join(p for p in co_parts if p).strip()
    ev_num = (
        [safe_float(s.get("heat", 0)), safe_float(s.get("event_intensity", 0)),
         safe_float(s.get("influence_range", 0)), safe_float(s.get("attribute_score", 0))]
        + one_hot(s.get("event_type", "未知"), cat_map["event_types"])
        + one_hot(s.get("duration_type", "未知"), cat_map["duration_types"])
    )
    assoc = safe_float(s.get("relation_score", 0.0))
    co_num = (
        one_hot(company.get("industry_lv1", "未知"), cat_map["industry_types"])
        + multi_hot(company.get("industry_chain_position", []) or ["未知"], cat_map["chain_positions"])
        + [assoc]
    )
    seq = s.get("input_window_pre2", s.get("price_sequence_pre2", []))
    rows = [seq[0], seq[1]] if len(seq) >= 2 else [{f: 0.0 for f in PRICE_FIELDS}] * 2
    return {
        "event_text": ev_text,
        "company_text": co_text,
        "event_num": ev_num,
        "company_num": co_num,
        "time_seq": [extract_price(rows[0]), extract_price(rows[1])],
        "delta_feat": compute_delta(rows),
        "true_up": s.get("targets", {}).get("future_4day_up"),
        "true_return": s.get("targets", {}).get("future_4day_return"),
        "stock_code": code,
        "stock_name": company.get("stock_name", code),
        "relation_score": assoc,
        "event_id": s.get("event_id", ""),
    }


# ========== 加载验证数据 + K=15 选取 ==========
print("=" * 60)
print(f"K={K} 验证集散点图生成")
print("=" * 60)

raw_val = json.loads(VAL_PATH.read_text(encoding="utf-8"))
val_samples = raw_val.get("samples", raw_val) if isinstance(raw_val, dict) else raw_val

# 按事件分组，每组取 relation_score 最高的 K 家
by_event = defaultdict(list)
for s in val_samples:
    by_event[s["event_id"]].append(s)

selected_samples = []
for eid, evts in by_event.items():
    evts_sorted = sorted(evts, key=lambda x: float(x.get("relation_score", 0)), reverse=True)
    selected_samples.extend(evts_sorted[:K])

print(f"  验证集原始: {len(val_samples)} 条")
print(f"  事件数: {len(by_event)}")
print(f"  K={K} 选取后: {len(selected_samples)} 条")

# 构建模型输入
raw = [build_sample(s) for s in selected_samples]
has_label = all(r["true_up"] is not None for r in raw)

# ========== 模型推理 ==========
ev_texts = ev_vocab.transform([s["event_text"] for s in raw])
co_texts = co_vocab.transform([s["company_text"] for s in raw])
ev_nums  = apply_std(np.array([s["event_num"] for s in raw], dtype=np.float32), scalers["event_num"])
co_nums  = apply_std(np.array([s["company_num"] for s in raw], dtype=np.float32), scalers["company_num"])
ts = np.array([[s["time_seq"][0], s["time_seq"][1]] for s in raw], dtype=np.float32)
flat_ts = apply_std(ts.reshape(-1, ts.shape[-1]), scalers["time_seq"])
ts = flat_ts.reshape(ts.shape).astype(np.float32)
df = apply_std(np.array([s["delta_feat"] for s in raw], dtype=np.float32), scalers["delta_feat"])

with torch.no_grad():
    out = net(
        torch.tensor(ev_texts, dtype=torch.float32),
        torch.tensor(ev_nums,  dtype=torch.float32),
        torch.tensor(co_texts, dtype=torch.float32),
        torch.tensor(co_nums,  dtype=torch.float32),
        torch.tensor(ts,       dtype=torch.float32),
        torch.tensor(df,      dtype=torch.float32),
    )
    fused = out["fused"].numpy()

# 分类头：pred_prob
z_cls  = fused @ W_c.T + b_c
Y_cls  = 1.0 / (1.0 + np.exp(-z_cls))  # ŷ_cls (预测上涨概率)

# 回归头：pred_return (从模型输出取)
reg_values = out["reg_value"].numpy()  # ŷ_reg (预测收益幅度)

# 真实标签
true_up     = np.array([1 if s["true_up"] else 0 for s in raw])
true_return = np.array([safe_float(s["true_return"]) for s in raw])

stock_names = [r["stock_name"] for r in raw]
n_samples = len(Y_cls)

print(f"  推理完成: {n_samples} 条")
print(f"  有标签: {has_label}")
print(f"  pred_prob 范围: [{Y_cls.min():.4f}, {Y_cls.max():.4f}]")
print(f"  pred_return 范围: [{reg_values.min():.4f}, {reg_values.max():.4f}]")
if has_label:
    acc = np.mean((Y_cls > 0.5).astype(int) == true_up)
    print(f"  方向准确率: {acc:.2%}")

# ========== 绘图 ==========
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "FangSong", "KaiTi"]
plt.rcParams["axes.unicode_minus"] = False

# 随机打乱顺序
rng = np.random.default_rng(42)
order = rng.permutation(n_samples)

# ---------- 图1: 概率散点图 ŷ_cls ----------
fig1, ax1 = plt.subplots(figsize=(14, 7))

for rank, i in enumerate(order):
    x = rank
    y_pred = float(Y_cls[i])
    y_true = float(true_up[i])
    name = stock_names[i]

    # 竖线连接（无箭头）
    gap = y_true - y_pred
    if abs(gap) > 0.03:
        ax1.plot([x, x], [y_pred, y_true], "-", color="#999999", lw=1.0, zorder=3)

    # 预测概率点（蓝色）
    ax1.plot(x, y_pred, "o", color="#1f77b4", markersize=8, zorder=4,
            markeredgecolor="#0d3b6e", markeredgewidth=0.8)
    # 实际概率点（红色）
    ax1.plot(x, y_true, "o", color="#d62728", markersize=8, zorder=4,
            markeredgecolor="#8b0000", markeredgewidth=0.8)

    # 标签：间隔标注
    if rank % 3 == 0 or abs(gap) > 0.5:
        ax1.text(x, y_pred - 0.06, f"{name} 预测:{y_pred:.2f}",
                ha="center", va="top", fontsize=6.5, color="#1f77b4",
                fontweight="bold")
        ax1.text(x, y_true + 0.04, f"实际:{y_true:.0f}",
                ha="center", va="bottom" if y_true == 1 else "top",
                fontsize=6.5, color="#d62728", fontweight="bold")

# 阈值线
ax1.axhline(0.5, color="black", linestyle="--", linewidth=1.5,
           label="threshold=0.5", zorder=2, alpha=0.6)

legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
           markersize=8, markeredgecolor="#0d3b6e", label=r"$\hat{Y}_{ij}^{cls}$ (预测)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
           markersize=8, markeredgecolor="#8b0000", label=r"$Y_{ij}^{cls}$ (实际)"),
    Line2D([0], [0], color="black", linestyle="--", lw=1.5, label="threshold=0.5"),
]
ax1.legend(handles=legend_elements, fontsize=10, loc="upper left",
          framealpha=0.9, edgecolor="#cccccc")

ax1.set_xlabel("样本序号（随机排列）", fontsize=12)
ax1.set_ylabel(r"$\hat{Y}_{ij}^{cls}$", fontsize=13)
ax1.set_title(r"test $\hat{Y}_{ij}^{cls}$ scatter (预测 vs 实际)", fontsize=13, fontweight="bold")
ax1.set_facecolor("white")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.15, 1.15)
ax1.set_xlim(-1, n_samples)
ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

plt.tight_layout()
out1 = Path(r"D:\Math_match\codes\outputs\task4\ycls_scatter_test.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close("all")
print(f"\n[概率图] {out1}")

# ---------- 图2: 幅度散点图 ŷ_reg ----------
fig2, ax2 = plt.subplots(figsize=(14, 7))

# 计算IQR边界（基于真实收益）
all_returns = true_return
q1_r = np.percentile(all_returns, 25)
q3_r = np.percentile(all_returns, 75)
iqr_r = q3_r - q1_r
lower_r = q1_r - 1.5 * iqr_r
upper_r = q3_r + 1.5 * iqr_r
mean_r = np.mean(all_returns)

for rank, i in enumerate(order):
    x = rank
    y_pred = float(reg_values[i])
    y_true = float(true_return[i])
    name = stock_names[i]

    # 竖线连接
    gap = y_true - y_pred
    if abs(gap) > 0.002:
        ax2.plot([x, x], [y_pred, y_true], "-", color="#999999", lw=1.0, zorder=3)

    # 预测收益点（蓝色）
    ax2.plot(x, y_pred, "o", color="#1f77b4", markersize=8, zorder=4,
            markeredgecolor="#0d3b6e", markeredgewidth=0.8)
    # 实际收益点（红色）
    ax2.plot(x, y_true, "o", color="#d62728", markersize=8, zorder=4,
            markeredgecolor="#8b0000", markeredgewidth=0.8)

    # 标签
    if rank % 3 == 0:
        ax2.text(x, y_pred - 0.006, f"{name} 预测:{y_pred:.4f}",
                ha="center", va="top", fontsize=6, color="#1f77b4", fontweight="bold")
        ax2.text(x, y_true + 0.004, f"实际:{y_true:.4f}",
                ha="center", va="bottom", fontsize=6, color="#d62728", fontweight="bold")

# 参考线
ax2.axhline(0, color="gray", linewidth=1, zorder=2)
ax2.axhline(mean_r, color="navy", linewidth=1.5, label=f"均值={mean_r:.4f}", zorder=2)
ax2.axhline(lower_r, color="red", linestyle="--", linewidth=1, alpha=0.7)
ax2.axhline(upper_r, color="red", linestyle="--", linewidth=1, alpha=0.7,
           label=f"IQR [{lower_r:.4f}, {upper_r:.4f}]")

legend_elements2 = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
           markersize=8, markeredgecolor="#0d3b6e", label=r"$\hat{Y}_{ij}^{reg}$ (预测)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
           markersize=8, markeredgecolor="#8b0000", label=r"$Y_{ij}^{reg}$ (实际)"),
    Line2D([0], [0], color="navy", lw=1.5, label=f"均值={mean_r:.4f}"),
    Line2D([0], [0], color="red", linestyle="--", lw=1, label=f"IQR [{lower_r:.4f}, {upper_r:.4f}]"),
]
ax2.legend(handles=legend_elements2, fontsize=9, loc="upper right",
          framealpha=0.9, edgecolor="#cccccc")

ax2.set_xlabel("样本序号（随机排列）", fontsize=12)
ax2.set_ylabel(r"$\hat{Y}_{ij}^{reg}$", fontsize=13)
ax2.set_title(r"test $\hat{Y}_{ij}^{reg}$ scatter (预测 vs 实际)", fontsize=13, fontweight="bold")
ax2.set_facecolor("white")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-1, n_samples)

plt.tight_layout()
out2 = Path(r"D:\Math_match\codes\outputs\task3\outlier_detection\car4_scatter_test.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close("all")
print(f"[幅度图] {out2}")

# ========== 汇总 ==========
print()
print("=" * 60)
print(f"K={K} 验证集散点图完成")
print("=" * 60)
print(f"  样本数: {n_samples}")
print(f"  概率图: {out1}")
print(f"  幅度图: {out2}")
if has_label:
    acc = np.mean((Y_cls > 0.5).astype(int) == true_up)
    mae = np.mean(np.abs(reg_values - true_return))
    print(f"  涨跌方向准确率: {acc:.2%}")
    print(f"  收益幅度 MAE: {mae:.4f}")
