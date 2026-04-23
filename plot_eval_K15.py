"""
K=15 验证集评估图（3张）
1）混淆矩阵图：pred_cls vs Y_cls（2×2 热力图）
2）ROC 曲线图：FPR vs TPR + AUC
3）真实值-预测值散点图：Y_reg vs ŷ_reg + y=x 参考线

数据来源：val_samples_v1.json（4.13事件），按 relation_score 取 Top-15 公司
"""
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

# ========== 路径 ==========
MODEL_DIR = Path(r"D:\Math_match\codes\outputs\task3\paper_training_lstm")
MODEL_PATH = MODEL_DIR / "best_model.pt"
ARTIFACTS = MODEL_DIR / "artifacts.pkl"
VAL_PATH = Path(r"D:\Math_match\codes\task3\dataset\val_samples_v1.json")
OUTPUT_DIR = Path(r"D:\Math_match\codes\outputs\task4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
K = 15


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

    def _tokenize(self, text):
        tokens = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            if len(text) < n:
                continue
            for i in range(len(text) - n + 1):
                tokens.append(text[i : i + n])
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
    time_input_dim=10,
    delta_dim=10,
    text_hidden_dim=64,
    num_hidden_dim=32,
    time_hidden_dim=32,
    fusion_hidden_dim=64,
    dropout=0.2,
)
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
net.load_state_dict(ckpt["model_state"], strict=False)
net.eval()

# ========== 加载公司画像 ==========
COMPANY_PROFILES_PATH = Path(r"D:\Math_match\codes\outputs\task2_profile_run\company_profiles.json")
company_payload = json.loads(COMPANY_PROFILES_PATH.read_text(encoding="utf-8"))
company_items = (
    company_payload.get("companies", company_payload)
    if isinstance(company_payload, dict)
    else company_payload
)
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


PRICE_FIELDS = [
    "open", "close", "high", "low", "volume", "amount",
    "amplitude", "pct_change", "change_amount", "turnover_rate",
]


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
    std = np.array(scaler["std"], dtype=np.float32)
    std[std < 1e-6] = 1.0
    return ((mat - mean) / std).astype(np.float32)


def build_sample(s):
    ev_text = s.get("event_features", {}).get("event_text", "") or ""
    if not ev_text:
        ev_text = f"{s.get('event_title', '')} {s.get('event_summary', '')}".strip()
    code = str(s.get("stock_code", "")).zfill(6)
    company = company_lk.get(code, {})
    co_parts = [
        company.get("stock_name", ""),
        company.get("summary_for_matching", ""),
        company.get("business_desc", ""),
    ]
    co_text = " ".join(p for p in co_parts if p).strip()
    ev_num = (
        [
            safe_float(s.get("heat", 0)),
            safe_float(s.get("event_intensity", 0)),
            safe_float(s.get("influence_range", 0)),
            safe_float(s.get("attribute_score", 0)),
        ]
        + one_hot(s.get("event_type", "未知"), cat_map["event_types"])
        + one_hot(s.get("duration_type", "未知"), cat_map["duration_types"])
    )
    assoc = safe_float(s.get("relation_score", 0.0))
    co_num = (
        one_hot(company.get("industry_lv1", "未知"), cat_map["industry_types"])
        + multi_hot(
            company.get("industry_chain_position", []) or ["未知"], cat_map["chain_positions"]
        )
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
    }


# ========== 加载数据 + K=15 选取 ==========
print("=" * 60)
print(f"K={K} 评估图生成（混淆矩阵 + ROC + 散点图）")
print("=" * 60)

raw_val = json.loads(VAL_PATH.read_text(encoding="utf-8"))
val_samples = raw_val.get("samples", raw_val) if isinstance(raw_val, dict) else raw_val

val_sorted = sorted(val_samples, key=lambda x: float(x.get("relation_score", 0)), reverse=True)
selected = val_sorted[:K]
print(f"  原始: {len(val_samples)} 条 -> K={K} 选取后: {len(selected)} 条")

raw = [build_sample(s) for s in selected]

# ========== 模型推理 ==========
ev_texts = ev_vocab.transform([s["event_text"] for s in raw])
co_texts = co_vocab.transform([s["company_text"] for s in raw])
ev_nums = apply_std(np.array([s["event_num"] for s in raw], dtype=np.float32), scalers["event_num"])
co_nums = apply_std(np.array([s["company_num"] for s in raw], dtype=np.float32), scalers["company_num"])
ts = np.array([[s["time_seq"][0], s["time_seq"][1]] for s in raw], dtype=np.float32)
flat_ts = apply_std(ts.reshape(-1, ts.shape[-1]), scalers["time_seq"])
ts = flat_ts.reshape(ts.shape).astype(np.float32)
df = apply_std(np.array([s["delta_feat"] for s in raw], dtype=np.float32), scalers["delta_feat"])

with torch.no_grad():
    out = net(
        torch.tensor(ev_texts, dtype=torch.float32),
        torch.tensor(ev_nums, dtype=torch.float32),
        torch.tensor(co_texts, dtype=torch.float32),
        torch.tensor(co_nums, dtype=torch.float32),
        torch.tensor(ts, dtype=torch.float32),
        torch.tensor(df, dtype=torch.float32),
    )
    fused = out["fused"].numpy()

# ŷ^{cls}（预测上涨概率）
z_cls = fused @ W_c.T + b_c
y_prob = 1.0 / (1.0 + np.exp(-z_cls)).flatten()

# ŷ^{reg}（预测收益幅度）
y_pred_reg = out["reg_value"].numpy().flatten()

# Y^{cls}（真实是否上涨）
Y_cls = np.array([1 if s["true_up"] else 0 for s in raw], dtype=int)

# Y^{reg}（真实收益）
Y_reg = np.array([safe_float(s["true_return"]) for s in raw], dtype=np.float64)

# pred_cls（按0.5阈值转成预测类别）
pred_cls = (y_prob >= 0.5).astype(int)

print(f"  推理完成: {len(raw)} 条")
print(f"  Y_cls 分布: 涨={sum(Y_cls)}, 不涨={sum(Y_cls == 0)}")
print(f"  pred_cls 分布: 涨={sum(pred_cls)}, 不涨={sum(pred_cls == 0)}")

# ========== 绘图 ==========
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "FangSong", "KaiTi"]
plt.rcParams["axes.unicode_minus"] = False


# ==================== 图1：混淆矩阵 ====================
cm = confusion_matrix(Y_cls, pred_cls)
# cm 形状: [[TN, FP], [FN, TP]]

fig1, ax1 = plt.subplots(figsize=(6, 5))
fig1.patch.set_facecolor("white")

labels_text = [["TN", "FP"], ["FN", "TP"]]
im = ax1.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax1.set_title(r"Confusion Matrix ($\hat{Y}^{cls}$ vs $Y^{cls}$)", fontsize=13, fontweight="bold")
ax1.set_xlabel(r"预测 $\hat{Y}^{cls}$", fontsize=12)
ax1.set_ylabel(r"真实 $Y^{cls}$", fontsize=12)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["0 (不涨)", "1 (涨)"])
ax1.set_yticks([0, 1])
ax1.set_yticklabels(["0 (不涨)", "1 (涨)"])

for i in range(2):
    for j in range(2):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax1.text(
            j, i, f"{labels_text[i][j]}={cm[i, j]}",
            ha="center", va="center", fontsize=14, fontweight="bold", color=color,
        )

fig1.colorbar(im, ax=ax1, shrink=0.8)
plt.tight_layout()
out1 = OUTPUT_DIR / "confusion_matrix.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close("all")
print(f"\n[混淆矩阵] {out1}")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")


# ==================== 图2：ROC 曲线 ====================
fpr, tpr, thresholds = roc_curve(Y_cls, y_prob)
auc_value = auc(fpr, tpr)

fig2, ax2 = plt.subplots(figsize=(7, 6))
fig2.patch.set_facecolor("white")

ax2.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=rf"ROC (AUC = {auc_value:.4f})")
ax2.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="随机基准")
ax2.fill_between(fpr, tpr, alpha=0.15, color="#1f77b4")

ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel("FPR (假正率)", fontsize=12)
ax2.set_ylabel("TPR (真正率)", fontsize=12)
ax2.set_title(r"ROC Curve ($\hat{Y}^{cls}$)", fontsize=13, fontweight="bold")
ax2.legend(loc="lower right", fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out2 = OUTPUT_DIR / "roc_curve.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close("all")
print(f"\n[ROC曲线] {out2}")
print(f"  AUC = {auc_value:.4f}")


# ==================== 图3：真实值-预测值散点图 ====================
fig3, ax3 = plt.subplots(figsize=(7, 6))
fig3.patch.set_facecolor("white")

ax3.scatter(
    Y_reg, y_pred_reg,
    c="#1f77b4", s=60, alpha=0.8,
    edgecolors="#0d3b6e", linewidths=0.8, zorder=3,
)

# y = x 参考线
all_vals = np.concatenate([Y_reg, y_pred_reg])
vmin, vmax = all_vals.min(), all_vals.max()
margin = (vmax - vmin) * 0.1
ax3.plot(
    [vmin - margin, vmax + margin],
    [vmin - margin, vmax + margin],
    color="red", lw=1.5, linestyle="--",
    label=r"$y = x$ 参考线", zorder=2,
)

ax3.set_xlabel(r"真实 $Y_{ij}^{reg}$", fontsize=12)
ax3.set_ylabel(r"预测 $\hat{Y}_{ij}^{reg}$", fontsize=12)
ax3.set_title(r"$Y_{ij}^{reg}$ vs $\hat{Y}_{ij}^{reg}$ scatter", fontsize=13, fontweight="bold")
ax3.legend(fontsize=11, loc="upper left")
ax3.grid(True, alpha=0.3)
ax3.set_xlim(vmin - margin, vmax + margin)
ax3.set_ylim(vmin - margin, vmax + margin)
ax3.set_aspect("equal", adjustable="box")

plt.tight_layout()
out3 = OUTPUT_DIR / "reg_scatter.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight", facecolor="white")
plt.close("all")
print(f"\n[散点图] {out3}")

mae = np.mean(np.abs(y_pred_reg - Y_reg))
rmse = np.sqrt(np.mean((y_pred_reg - Y_reg) ** 2))
print(f"  MAE = {mae:.6f}")
print(f"  RMSE = {rmse:.6f}")

print(f"\n完成！输出目录: {OUTPUT_DIR}")
