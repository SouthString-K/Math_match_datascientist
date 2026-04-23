"""
绘制测试集 Y_cls 按实际标签分组的柱状图
"""
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

MODEL_DIR  = Path(r"D:\Math_match\codes\outputs\task3\paper_training_lstm")
MODEL_PATH = MODEL_DIR / "best_model.pt"
ARTIFACTS  = MODEL_DIR / "artifacts.pkl"
ENRICHED   = MODEL_DIR / "enriched_dataset.json"
TEST_PATH  = Path(r"D:\Math_match\codes\task3\dataset\test_samples_v2_custom_414.json")

import math

class CharTfidfVectorizer:
    def __init__(self, max_features=2000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary_ = {}
        self.idf_ = None

    def _tokenize(self, text):
        tokens = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            if len(text) < n:
                continue
            for i in range(len(text) - n + 1):
                tokens.append(text[i:i+n])
        return tokens

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
            self.idf_[i] = math.log((1+n)/(1+df.get(t,1))) + 1.0
        return self

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

head = json.loads((MODEL_DIR / "head_params.json").read_text(encoding="utf-8"))
W_c, b_c = np.array(head["W_c"]), np.array(head["b_c"])

enr = json.loads(ENRICHED.read_text(encoding="utf-8"))
train_samples = enr["train"]

ev_vocab = CharTfidfVectorizer(max_features=128)
ev_vocab.fit([s["event_text"] for s in train_samples])
co_vocab = CharTfidfVectorizer(max_features=192)
co_vocab.fit([s["company_text"] for s in train_samples])

with open(ARTIFACTS, "rb") as f:
    art = pickle.load(f)
scalers = art["scalers"]
cat_map = art["category_map"]

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

raw_test = json.loads(TEST_PATH.read_text(encoding="utf-8"))
raw_test = raw_test if isinstance(raw_test, list) else raw_test.get("samples", raw_test)

from data_loader import load_company_lookup, load_stock_history
company_lk = load_company_lookup()

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

PRICE_FIELDS = ["open","close","high","low","volume","amount",
                 "amplitude","pct_change","change_amount","turnover_rate"]

def extract_price(row):
    if row is None:
        return [0.0] * len(PRICE_FIELDS)
    return [safe_float(row.get(f, 0)) for f in PRICE_FIELDS]

def compute_delta(rows):
    if len(rows) < 2:
        return [0.0] * len(PRICE_FIELDS)
    a, b = np.array(extract_price(rows[0]), dtype=np.float32), \
           np.array(extract_price(rows[1]), dtype=np.float32)
    return (b - a).tolist()

def apply_std(mat, scaler):
    mean = np.array(scaler["mean"], dtype=np.float32)
    std  = np.array(scaler["std"],  dtype=np.float32)
    std[std < 1e-6] = 1.0
    return ((mat - mean) / std).astype(np.float32)

def build_test_sample(s):
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
    seq = s.get("price_sequence_pre2", [])
    rows = [seq[0], seq[1]] if len(seq) >= 2 else [{f: 0.0 for f in PRICE_FIELDS}] * 2
    return {
        "event_text": ev_text,
        "company_text": co_text,
        "event_num": ev_num,
        "company_num": co_num,
        "time_seq": [extract_price(rows[0]), extract_price(rows[1])],
        "delta_feat": compute_delta(rows),
        "true_up": s.get("targets", {}).get("future_4day_up"),
        "stock_code": code,
        "stock_name": company.get("stock_name", code),
    }

raw = [build_test_sample(s) for s in raw_test]

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

z_cls  = fused @ W_c.T + b_c
Y_cls  = 1.0 / (1.0 + np.exp(-z_cls))
true_labels = np.array([1 if s["true_up"] else 0 for s in raw])

# ---------- 绘图 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "FangSong", "KaiTi"]
plt.rcParams["axes.unicode_minus"] = False

neg_mask = true_labels == 0
pos_mask = true_labels == 1
neg_mean = float(Y_cls[neg_mask].mean())
pos_mean = float(Y_cls[pos_mask].mean())

fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.bar(
    ["Neg\n(Future Return < 0)\nn={}".format(neg_mask.sum()),
     "Pos\n(Future Return > 0)\nn={}".format(pos_mask.sum())],
    [neg_mean, pos_mean],
    color=["#2ca02c", "#d62728"],
    width=0.5,
    edgecolor="white",
    linewidth=1.2,
)

# 在柱顶标注数值
for bar, val in zip(bars, [neg_mean, pos_mean]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_ylabel(r"$\hat{Y}_{ij}^{cls}$ Mean (probability)", fontsize=12)
ax.set_title(r"test $\hat{Y}_{ij}^{cls}$ by True Label", fontsize=13, fontweight="bold")
ax.set_facecolor("white")
ax.grid(True, axis="y", alpha=0.3)
ax.set_ylim(0, max(neg_mean, pos_mean) * 1.3)

plt.tight_layout()
out_path = r"D:\Math_match\codes\outputs\task4\ycls_bar_test.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close("all")
print(f"Saved: {out_path}")

print(f"\nTest Y_cls:")
print(f"  Neg: mean={neg_mean:.4f} (n={neg_mask.sum()})")
print(f"  Pos: mean={pos_mean:.4f} (n={pos_mask.sum()})")
