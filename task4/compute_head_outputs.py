"""
用已训练好的模型参数 (W_c, b_c, W_r, b_r) 计算 Task4 各窗口样本的：
  Y_cls_ij = sigmoid(W_c @ f_ij + b_c)
  Y_reg_ij = W_r @ f_ij + b_r

f_ij = 模型融合层输出的 64 维向量
"""
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

# ---------- 路径 ----------
MODEL_DIR = Path(r"D:\Math_match\codes\outputs\task3\paper_training_lstm")
MODEL_PATH = MODEL_DIR / "best_model.pt"
ARTIFACTS_PATH = MODEL_DIR / "artifacts.pkl"
HEAD_PARAMS_PATH = MODEL_DIR / "head_params.json"

# ---------- 加载 head 参数 ----------
head_params = json.loads(HEAD_PARAMS_PATH.read_text(encoding="utf-8"))
W_c = np.array(head_params["W_c"])   # (1, 64)
b_c = np.array(head_params["b_c"])   # (1,)
W_r = np.array(head_params["W_r"])   # (1, 64)
b_r = np.array(head_params["b_r"])   # (1,)

print(f"W_c shape: {W_c.shape}, b_c: {b_c}")
print(f"W_r shape: {W_r.shape}, b_r: {b_r}")


# ---------- TF-IDF（与 inference.py 完全一致） ----------
class CharTfidfVectorizer:
    def __init__(self, max_features=2000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary_ = {}
        self.idf_ = None

    def _tokenize(self, text):
        tokens = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            if len(text) < n:
                continue
            for idx in range(len(text) - n + 1):
                tokens.append(text[idx: idx + n])
        return tokens

    def fit(self, texts):
        doc_freq = {}
        for text in texts:
            seen = set(self._tokenize(text))
            for t in seen:
                doc_freq[t] = doc_freq.get(t, 0) + 1
        sorted_tokens = sorted(doc_freq.items(), key=lambda x: (-x[1], x[0]))[:self.max_features]
        self.vocabulary_ = {t: i for i, (t, _) in enumerate(sorted_tokens)}
        n_docs = max(len(texts), 1)
        self.idf_ = np.ones(len(self.vocabulary_), dtype=np.float32)
        for t, idx in self.vocabulary_.items():
            df = doc_freq.get(t, 1)
            self.idf_[idx] = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        return self

    def transform(self, texts):
        n = len(self.vocabulary_)
        matrix = np.zeros((len(texts), n), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            tf = {}
            for t in self._tokenize(text):
                if t in self.vocabulary_:
                    idx = self.vocabulary_[t]
                    tf[idx] = tf.get(idx, 0) + 1
            for idx, cnt in tf.items():
                matrix[row_idx, idx] = float(cnt) * self.idf_[idx]
            norm = np.linalg.norm(matrix[row_idx])
            if norm > 0:
                matrix[row_idx] /= norm
        return matrix


def safe_float(v, default=0.0):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except:
        return default


def build_event_text(event):
    return f"{event.get('title', '')} {event.get('event_summary', '')}".strip()


def build_company_text(company):
    parts = [company.get("stock_name", ""),
             company.get("summary_for_matching", ""),
             company.get("business_desc", "")]
    return " ".join(p for p in parts if p).strip()


def one_hot(v, cats):
    return [1.0 if v == c else 0.0 for c in cats]


def multi_hot(vs, cats):
    s = set(vs or [])
    return [1.0 if c in s else 0.0 for c in cats]


PRICE_FIELDS = ["open", "close", "high", "low", "volume", "amount",
                 "amplitude", "pct_change", "change_amount", "turnover_rate"]


def extract_price_vector(row):
    if row is None:
        return [0.0] * len(PRICE_FIELDS)
    return [safe_float(row.get(f, 0)) for f in PRICE_FIELDS]


def compute_delta(rows):
    if len(rows) < 2:
        return [0.0] * len(PRICE_FIELDS)
    a = np.array(extract_price_vector(rows[0]), dtype=np.float32)
    b = np.array(extract_price_vector(rows[1]), dtype=np.float32)
    return (b - a).tolist()


def apply_standardizer(mat, scaler):
    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)
    std[std < 1e-6] = 1.0
    return ((mat - mean) / std).astype(np.float32)


# ---------- 加载数据和模型 ----------
print("\n加载数据...")

from data_loader import (
    load_events, load_company_lookup, load_assoc_matrix, load_stock_history,
    build_windows, filter_events, build_stock_pool,
)

events     = load_events()
assoc      = load_assoc_matrix()
company_lk = load_company_lookup()
history    = load_stock_history()
windows    = build_windows()

with open(ARTIFACTS_PATH, "rb") as f:
    artifacts = pickle.load(f)
scalers    = artifacts["scalers"]
cat_map    = artifacts["category_map"]

# 加载模型
import importlib.util
spec = importlib.util.spec_from_file_location("_m", Path(r"D:\Math_match\codes\task3\model.py"))
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
PaperEventLSTM = model_module.PaperEventLSTM

n_industry = len(cat_map["industry_types"])
n_chain    = len(cat_map["chain_positions"])

net = PaperEventLSTM(
    event_text_dim=128,
    event_num_dim=4 + len(cat_map["event_types"]) + len(cat_map["duration_types"]),
    company_text_dim=192,
    company_num_dim=n_industry + n_chain + 1,
    time_input_dim=10, delta_dim=10,
    text_hidden_dim=64, num_hidden_dim=32,
    time_hidden_dim=32, fusion_hidden_dim=64, dropout=0.2,
)
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
net.load_state_dict(ckpt["model_state"], strict=False)
net.eval()
print("模型已加载")

# ---------- 对三个窗口逐个计算 f_ij → Y_cls / Y_reg ----------
print("\n" + "=" * 60)
print("各窗口 Y_cls / Y_reg 计算结果")
print("=" * 60)

all_results = []

for window in windows:
    wid = window["window_id"]
    print(f"\n--- 窗口{wid} ---")

    filtered = filter_events(events, window["publish_times"])
    if not filtered:
        print("  无有效事件")
        continue

    stock_pool = build_stock_pool(filtered, assoc, company_lk)
    if not stock_pool:
        print("  无候选股票")
        continue

    # 构造推理样本（原始特征）
    raw_samples = []
    tau_date = None
    for evt in filtered:
        tau_date = evt.get("publish_time", "")[:8]
        for item in stock_pool:
            code   = item["stock_code"]
            assoc_val = item["assoc_ij"]
            company = company_lk.get(code, {})

            code_history = history.get(code, [])
            all_dates = sorted(r["date"] for r in code_history)
            event_dates = [d for d in all_dates if d < tau_date]
            pre2 = event_dates[-2:] if len(event_dates) >= 2 else event_dates

            rows = []
            for d in pre2:
                row = next((r for r in code_history if r["date"] == d), None)
                rows.append(row if row else {f: 0.0 for f in PRICE_FIELDS})
            while len(rows) < 2:
                rows.insert(0, {f: 0.0 for f in PRICE_FIELDS})

            raw_samples.append({
                "event": evt,
                "stock_code": code,
                "stock_name": item["stock_name"],
                "assoc_ij": assoc_val,
                "event_text": build_event_text(evt),
                "company_text": build_company_text(company),
                "event_num": (
                    [safe_float(evt.get("heat", 0)),
                     safe_float(evt.get("event_intensity", 0)),
                     safe_float(evt.get("influence_range", 0)),
                     safe_float(evt.get("attribute_score", 0))]
                    + one_hot(evt.get("event_type", "未知"), cat_map["event_types"])
                    + one_hot(evt.get("duration_type", "未知"), cat_map["duration_types"])
                ),
                "company_num": (
                    one_hot(company.get("industry_lv1", "未知"), cat_map["industry_types"])
                    + multi_hot(company.get("industry_chain_position", []) or ["未知"], cat_map["chain_positions"])
                    + [safe_float(assoc_val)]
                ),
                "time_seq": [extract_price_vector(rows[0]), extract_price_vector(rows[1])],
                "delta_feat": compute_delta(rows),
            })

    if not raw_samples:
        continue

    # TF-IDF
    ev_vec = CharTfidfVectorizer(max_features=128)
    ev_vec.fit([s["event_text"] for s in raw_samples])
    ev_texts = ev_vec.transform([s["event_text"] for s in raw_samples])

    co_vec = CharTfidfVectorizer(max_features=192)
    co_vec.fit([s["company_text"] for s in raw_samples])
    co_texts = co_vec.transform([s["company_text"] for s in raw_samples])

    ev_nums  = apply_standardizer(np.array([s["event_num"] for s in raw_samples], dtype=np.float32), scalers["event_num"])
    co_nums  = apply_standardizer(np.array([s["company_num"] for s in raw_samples], dtype=np.float32), scalers["company_num"])
    ts = np.array([[s["time_seq"][0], s["time_seq"][1]] for s in raw_samples], dtype=np.float32)
    flat_ts = ts.reshape(-1, ts.shape[-1])
    flat_ts = apply_standardizer(flat_ts, scalers["time_seq"])
    ts = flat_ts.reshape(ts.shape).astype(np.float32)
    df = apply_standardizer(np.array([s["delta_feat"] for s in raw_samples], dtype=np.float32), scalers["delta_feat"])

    # 推理，提取 fused 特征
    with torch.no_grad():
        out = net(
            torch.tensor(ev_texts, dtype=torch.float32),
            torch.tensor(ev_nums,  dtype=torch.float32),
            torch.tensor(co_texts, dtype=torch.float32),
            torch.tensor(co_nums,  dtype=torch.float32),
            torch.tensor(ts,       dtype=torch.float32),
            torch.tensor(df,      dtype=torch.float32),
        )
        fused = out["fused"].numpy()   # (N, 64)

    # 用 W_c / b_c / W_r / b_r 手动算 Y
    z_cls = fused @ W_c.T + b_c   # (N, 1)
    z_reg = fused @ W_r.T + b_r   # (N, 1)
    Y_cls = 1.0 / (1.0 + np.exp(-z_cls))  # sigmoid
    Y_reg = z_reg.copy()  # 回归头无激活函数，直接输出

    print(f"  候选样本数: {len(raw_samples)}")
    print(f"  fused 维度: {fused.shape}")
    print()
    print(f"  {'股票代码':<8} {'股票名称':<8} {'fused norm':>12} {'z_cls':>10} {'Y_cls':>8} {'z_reg':>10} {'Y_reg':>10}")
    print(f"  {'-'*70}")

    window_preds = []
    for i, s in enumerate(raw_samples):
        row = {
            "window_id": wid,
            "stock_code": s["stock_code"],
            "stock_name": s["stock_name"],
            "assoc_ij": s["assoc_ij"],
            "fused_norm": float(np.linalg.norm(fused[i])),
            "z_cls": float(z_cls[i, 0]),
            "Y_cls": float(Y_cls[i, 0]),
            "z_reg": float(z_reg[i, 0]),
            "Y_reg": float(Y_reg[i, 0]),
        }
        window_preds.append(row)
        print(f"  {row['stock_code']:<8} {row['stock_name']:<8} "
              f"{row['fused_norm']:>12.4f} {row['z_cls']:>10.4f} {row['Y_cls']:>8.4f} "
              f"{row['z_reg']:>10.4f} {row['Y_reg']:>10.4f}")

    all_results.extend(window_preds)

# 保存
out_path = Path(r"D:\Math_match\codes\outputs\task4\fused_features_Y_values.json")
out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n已保存: {out_path}")
