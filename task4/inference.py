"""
Task 4: task3 推理模块
- 构建推理样本（char TF-IDF + 数值特征 + 双交易日价格序列）
- 加载真实模型 best_model.pt + artifacts.pkl，执行批量推理
"""
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

from data_loader import (
    parse_event_date, get_stock_row,
    safe_float,
)

# ---------- 路径 ----------
MODEL_DIR = Path(r"D:\Math_match\codes\outputs\task3\paper_training_lstm")
MODEL_PATH = MODEL_DIR / "best_model.pt"
ARTIFACTS_PATH = MODEL_DIR / "artifacts.pkl"

# ---------- 类别编码（与训练完全一致） ----------
EVENT_TYPE_CLASSES = ["政策类", "宏观类", "行业类", "公司类", "地缘类"]
DURATION_CLASSES = ["脉冲型", "长尾型", "持续型"]

# industry_types / chain_positions 动态从 artifacts 加载


# ---------- TF-IDF ----------
class CharTfidfVectorizer:
    def __init__(self, max_features: int = 2000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary_ = {}
        self.idf_ = None

    @classmethod
    def from_vocabulary(cls, vocabulary: dict, ngram_range=(1, 2)):
        """从训练时保存的词汇表初始化，保证维度与训练一致"""
        obj = cls(max_features=len(vocabulary), ngram_range=ngram_range)
        obj.vocabulary_ = dict(vocabulary)
        obj.idf_ = np.ones(len(vocabulary), dtype=np.float32)
        return obj

    def _tokenize(self, text: str):
        text = text or ""
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
            for token in seen:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        sorted_tokens = sorted(doc_freq.items(), key=lambda x: (-x[1], x[0]))[:self.max_features]
        self.vocabulary_ = {t: i for i, (t, _) in enumerate(sorted_tokens)}
        n_docs = max(len(texts), 1)
        self.idf_ = np.ones(len(self.vocabulary_), dtype=np.float32)
        for token, idx in self.vocabulary_.items():
            df = doc_freq.get(token, 1)
            self.idf_[idx] = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        return self

    def transform(self, texts):
        n = len(self.vocabulary_)
        matrix = np.zeros((len(texts), n), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            tf = {}
            for token in self._tokenize(text):
                if token in self.vocabulary_:
                    idx = self.vocabulary_[token]
                    tf[idx] = tf.get(idx, 0) + 1
            for idx, cnt in tf.items():
                matrix[row_idx, idx] = float(cnt) * self.idf_[idx]
            norm = np.linalg.norm(matrix[row_idx])
            if norm > 0:
                matrix[row_idx] /= norm
        return matrix


def one_hot(value, categories):
    return [1.0 if value == c else 0.0 for c in categories]


def multi_hot(values, categories):
    value_set = set(values or [])
    return [1.0 if c in value_set else 0.0 for c in categories]


# ---------- 特征构建 ----------
PRICE_FIELDS = ["open", "close", "high", "low", "volume", "amount",
                "amplitude", "pct_change", "change_amount", "turnover_rate"]


def build_event_text(event: dict) -> str:
    title = event.get("title", "") or ""
    summary = event.get("event_summary", "") or ""
    return f"{title} {summary}".strip()


def build_company_text(company: dict) -> str:
    parts = [
        company.get("stock_name", ""),
        company.get("summary_for_matching", ""),
        company.get("business_desc", ""),
    ]
    return " ".join(p for p in parts if p).strip()


def build_event_num(event: dict, category_map: dict) -> list:
    base = [
        safe_float(event.get("heat", 0)),
        safe_float(event.get("event_intensity", 0)),
        safe_float(event.get("influence_range", 0)),
        safe_float(event.get("attribute_score", 0)),
    ]
    return (
        base
        + one_hot(event.get("event_type", "未知"), category_map["event_types"])
        + one_hot(event.get("duration_type", "未知"), category_map["duration_types"])
    )


def build_company_num(company: dict, assoc_score: float, category_map: dict) -> list:
    industry_vec = one_hot(company.get("industry_lv1", "未知"), category_map["industry_types"])
    chain_vec = multi_hot(
        company.get("industry_chain_position", []) or ["未知"],
        category_map["chain_positions"]
    )
    return industry_vec + chain_vec + [safe_float(assoc_score)]


def extract_price_vector(day_row) -> list:
    if day_row is None:
        return [0.0] * len(PRICE_FIELDS)
    return [safe_float(day_row.get(f, 0)) for f in PRICE_FIELDS]


def compute_delta(window_rows) -> list:
    if len(window_rows) < 2:
        return [0.0] * len(PRICE_FIELDS)
    first = np.array(extract_price_vector(window_rows[0]), dtype=np.float32)
    second = np.array(extract_price_vector(window_rows[1]), dtype=np.float32)
    return (second - first).tolist()


# ---------- Standardizer ----------
def apply_standardizer(matrix, scaler):
    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)
    std[std < 1e-6] = 1.0
    return ((matrix - mean) / std).astype(np.float32)


# ---------- 推理样本构造 ----------
def build_inference_samples(window_id: int, events: list, stock_pool: list,
                            company_lookup: dict, history: dict,
                            category_map: dict):
    """
    为窗口内所有 (事件, 股票) 构造推理样本原始特征（未标准化）
    """
    samples = []
    for evt in events:
        sid = evt["sample_id"]
        tau_date = parse_event_date(evt["publish_time"])

        for item in stock_pool:
            code = item["stock_code"]
            assoc_val = item["assoc_ij"]

            # 获取事件前两个交易日
            code_history = history.get(code, [])
            all_dates = sorted(r["date"] for r in code_history)
            event_dates = [d for d in all_dates if d < tau_date]
            pre2_dates = event_dates[-2:] if len(event_dates) >= 2 else event_dates

            window_rows = []
            for d in pre2_dates:
                row = get_stock_row(history, code, d)
                window_rows.append(row if row else {f: 0.0 for f in PRICE_FIELDS})
            while len(window_rows) < 2:
                window_rows.insert(0, {f: 0.0 for f in PRICE_FIELDS})

            time_seq = [extract_price_vector(window_rows[0]), extract_price_vector(window_rows[1])]
            delta_feat = compute_delta(window_rows)

            company = company_lookup.get(code, {})

            samples.append({
                "window_id": window_id,
                "event_id": sid,
                "publish_time": evt["publish_time"],
                "event_type": evt.get("event_type", ""),
                "stock_code": code,
                "stock_name": item["stock_name"],
                "assoc_ij": assoc_val,
                "event_text": build_event_text(evt),
                "company_text": build_company_text(company),
                "event_num": build_event_num(evt, category_map),
                "company_num": build_company_num(company, assoc_val, category_map),
                "time_seq": time_seq,
                "delta_feat": delta_feat,
                "tau_date": tau_date,
            })
    return samples


# ---------- 模型加载 ----------
def load_artifacts():
    with open(ARTIFACTS_PATH, "rb") as f:
        return pickle.load(f)


def load_model():
    """加载 PaperEventLSTM"""
    if not MODEL_PATH.exists():
        print(f"[Inference] 模型不存在: {MODEL_PATH}")
        return None

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_PaperEventLSTM_module",
        Path(r"D:\Math_match\codes\task3\model.py")
    )
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    _PaperEventLSTM = model_module.PaperEventLSTM

    artifacts = load_artifacts()
    cat_map = artifacts["category_map"]
    n_industry = len(cat_map["industry_types"])
    n_chain = len(cat_map["chain_positions"])

    # 构建模型（维度与训练一致）
    model = _PaperEventLSTM(
        event_text_dim=128,
        event_num_dim=4 + len(cat_map["event_types"]) + len(cat_map["duration_types"]),  # 4 + 3 + 2 = 9
        company_text_dim=192,
        company_num_dim=n_industry + n_chain + 1,  # 动态
        time_input_dim=10,
        delta_dim=10,
        text_hidden_dim=64,
        num_hidden_dim=32,
        time_hidden_dim=32,
        fusion_hidden_dim=64,
        dropout=0.2,
    )

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    print(f"[Inference] 模型已加载 (best_val_auc={ckpt.get('best_val_auc','N/A')})")
    return model, artifacts


# ---------- 批量推理 ----------
@torch.no_grad()
def run_inference(samples_raw, model_artifacts):
    """
    samples_raw: build_inference_samples 的原始输出
    返回预测结果列表
    """
    if not samples_raw:
        return []

    model, artifacts = model_artifacts
    scalers = artifacts["scalers"]

    # TF-IDF：优先使用训练时保存的词汇表，保证维度一致
    if "event_vocabulary" in artifacts and "company_vocabulary" in artifacts:
        event_vectorizer = CharTfidfVectorizer.from_vocabulary(
            artifacts["event_vocabulary"], ngram_range=(1, 2)
        )
        company_vectorizer = CharTfidfVectorizer.from_vocabulary(
            artifacts["company_vocabulary"], ngram_range=(1, 2)
        )
    else:
        event_vectorizer = CharTfidfVectorizer(max_features=128, ngram_range=(1, 2))
        event_vectorizer.fit([s["event_text"] for s in samples_raw])
        company_vectorizer = CharTfidfVectorizer(max_features=192, ngram_range=(1, 2))
        company_vectorizer.fit([s["company_text"] for s in samples_raw])

    event_texts = event_vectorizer.transform([s["event_text"] for s in samples_raw])
    company_texts = company_vectorizer.transform([s["company_text"] for s in samples_raw])

    # 数值特征
    event_nums = np.array([s["event_num"] for s in samples_raw], dtype=np.float32)
    company_nums = np.array([s["company_num"] for s in samples_raw], dtype=np.float32)
    time_seqs = np.array([[s["time_seq"][0], s["time_seq"][1]] for s in samples_raw], dtype=np.float32)
    delta_feats = np.array([s["delta_feat"] for s in samples_raw], dtype=np.float32)

    # 标准化
    event_nums = apply_standardizer(event_nums, scalers["event_num"])
    company_nums = apply_standardizer(company_nums, scalers["company_num"])
    flat_time = time_seqs.reshape(-1, time_seqs.shape[-1])
    flat_time = apply_standardizer(flat_time, scalers["time_seq"])
    time_seqs = flat_time.reshape(time_seqs.shape).astype(np.float32)
    delta_feats = apply_standardizer(delta_feats, scalers["delta_feat"])

    # to tensor
    event_text_t = torch.tensor(event_texts, dtype=torch.float32)
    event_num_t = torch.tensor(event_nums, dtype=torch.float32)
    company_text_t = torch.tensor(company_texts, dtype=torch.float32)
    company_num_t = torch.tensor(company_nums, dtype=torch.float32)
    time_seq_t = torch.tensor(time_seqs, dtype=torch.float32)
    delta_t = torch.tensor(delta_feats, dtype=torch.float32)

    # 模型推理
    out = model(event_text_t, event_num_t, company_text_t, company_num_t, time_seq_t, delta_t)
    cls_logits = out["cls_logit"].numpy()
    reg_values = out["reg_value"].numpy()

    # sigmoid
    probs = 1.0 / (1.0 + np.exp(-cls_logits))

    results = []
    for i, s in enumerate(samples_raw):
        results.append({
            "window_id": s["window_id"],
            "event_id": s["event_id"],
            "publish_time": s["publish_time"],
            "event_type": s["event_type"],
            "stock_code": s["stock_code"],
            "stock_name": s["stock_name"],
            "assoc_ij": s["assoc_ij"],
            "pred_prob": round(float(probs[i]), 4),
            "pred_return": round(float(reg_values[i]), 6),
        })
    return results
