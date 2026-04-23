import csv
import json
import math
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import PaperEventLSTM, count_params

sys.stdout.reconfigure(encoding="utf-8")

TRAIN_SAMPLES_PATH = Path(r"D:\Math_match\codes\task3\dataset\train_samples_v1.json")
VAL_SAMPLES_PATH = Path(r"D:\Math_match\codes\task3\dataset\val_samples_v1.json")
PENDING_SAMPLES_PATH = Path(r"D:\Math_match\codes\task3\dataset\pending_inference_samples_v1.json")
EVENTS_PATH = Path(r"D:\Math_match\codes\outputs\classification_run_eastmoney400_v2\classification_results_new.json")
COMPANY_PROFILES_PATH = Path(r"D:\Math_match\codes\outputs\task2_profile_run\company_profiles.json")
STOCK_HISTORY_PATHS = [
    Path(r"D:\Math_match\codes\task3\hs300_history_batch1.json"),
    Path(r"D:\Math_match\codes\task3\hs300_history_batch2.json"),
]
OUTPUT_DIR = Path(r"D:\Math_match\codes\outputs\task3\paper_training_lstm")

ALPHA = 1.0
BETA = 1.0
TOP_K_EFFECTIVENESS = 3
RANDOM_SEED = 42
EPOCHS = 60
PATIENCE = 12
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PRICE_FIELDS = [
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "amplitude",
    "pct_change",
    "change_amount",
    "turnover_rate",
]


class CharTfidfVectorizer:
    def __init__(self, max_features: int, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary_ = {}
        self.idf_ = None

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
        doc_freq = defaultdict(int)
        for text in texts:
            seen = set(self._tokenize(text))
            for token in seen:
                doc_freq[token] += 1

        sorted_tokens = sorted(
            doc_freq.items(),
            key=lambda item: (-item[1], item[0]),
        )[: self.max_features]

        self.vocabulary_ = {token: idx for idx, (token, _) in enumerate(sorted_tokens)}
        n_docs = max(len(texts), 1)
        self.idf_ = np.ones(len(self.vocabulary_), dtype=np.float32)
        for token, idx in self.vocabulary_.items():
            df = doc_freq[token]
            self.idf_[idx] = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        return self

    def transform(self, texts):
        matrix = np.zeros((len(texts), len(self.vocabulary_)), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            tf = defaultdict(int)
            for token in self._tokenize(text):
                if token in self.vocabulary_:
                    tf[token] += 1
            for token, count in tf.items():
                col_idx = self.vocabulary_[token]
                matrix[row_idx, col_idx] = float(count) * self.idf_[col_idx]
            norm = np.linalg.norm(matrix[row_idx])
            if norm > 0:
                matrix[row_idx] /= norm
        return matrix


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_stock_code(stock_code) -> str:
    return str(stock_code).strip().zfill(6)


def safe_float(value) -> float:
    if value is None:
        return 0.0
    try:
        if isinstance(value, str) and not value.strip():
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_stock_row(row: dict) -> dict:
    return {
        "date": str(row.get("日期", row.get("date"))),
        "stock_code": normalize_stock_code(row.get("股票代码", row.get("stock_code", ""))),
        "open": safe_float(row.get("开盘", row.get("open"))),
        "close": safe_float(row.get("收盘", row.get("close"))),
        "high": safe_float(row.get("最高", row.get("high"))),
        "low": safe_float(row.get("最低", row.get("low"))),
        "volume": safe_float(row.get("成交量", row.get("volume"))),
        "amount": safe_float(row.get("成交额", row.get("amount"))),
        "amplitude": safe_float(row.get("振幅", row.get("amplitude"))),
        "pct_change": safe_float(row.get("涨跌幅", row.get("pct_change"))),
        "change_amount": safe_float(row.get("涨跌额", row.get("change_amount"))),
        "turnover_rate": safe_float(row.get("换手率", row.get("turnover_rate"))),
    }


def load_stock_history(paths):
    merged = defaultdict(dict)
    for path in paths:
        payload = load_json(path)
        for stock_code, rows in payload.get("data_by_stock", {}).items():
            norm_code = normalize_stock_code(stock_code)
            for row in rows:
                normalized = normalize_stock_row(row)
                merged[norm_code][normalized["date"]] = normalized

    sorted_history = {}
    for stock_code, rows_by_date in merged.items():
        dates = sorted(rows_by_date.keys())
        sorted_history[stock_code] = [rows_by_date[d] for d in dates]
    return sorted_history


def compute_daily_returns(stock_history):
    stock_returns = {}
    benchmark_bucket = defaultdict(list)

    for stock_code, rows in stock_history.items():
        returns = {}
        for idx in range(1, len(rows)):
            prev_close = safe_float(rows[idx - 1]["close"])
            close = safe_float(rows[idx]["close"])
            if prev_close <= 0:
                continue
            daily_return = (close - prev_close) / prev_close
            date = rows[idx]["date"]
            returns[date] = daily_return
            benchmark_bucket[date].append(daily_return)
        stock_returns[stock_code] = returns

    benchmark_returns = {
        date: float(np.mean(values))
        for date, values in benchmark_bucket.items()
        if values
    }
    return stock_returns, benchmark_returns


def load_event_lookup():
    events = load_json(EVENTS_PATH)
    return {event["sample_id"]: event for event in events}


def load_company_lookup():
    payload = load_json(COMPANY_PROFILES_PATH)
    companies = payload.get("companies", payload) if isinstance(payload, dict) else payload
    return {normalize_stock_code(company["stock_code"]): company for company in companies}


def collect_categories(samples, event_lookup, company_lookup):
    event_types = set()
    duration_types = set()
    industry_types = set()
    chain_positions = set()

    for sample in samples:
        event = event_lookup[sample["event_id"]]
        company = company_lookup.get(normalize_stock_code(sample["stock_code"]))
        event_types.add(event.get("event_type", "未知"))
        duration_types.add(event.get("duration_type", "未知"))
        if company:
            industry_types.add(company.get("industry_lv1", "未知"))
            for chain in company.get("industry_chain_position", []) or ["未知"]:
                chain_positions.add(chain)

    return {
        "event_types": sorted(event_types),
        "duration_types": sorted(duration_types),
        "industry_types": sorted(industry_types),
        "chain_positions": sorted(chain_positions),
    }


def one_hot(value, categories):
    return [1.0 if value == category else 0.0 for category in categories]


def multi_hot(values, categories):
    value_set = set(values or [])
    return [1.0 if category in value_set else 0.0 for category in categories]


def build_event_text(sample, event):
    title = event.get("title") or sample.get("event_title", "")
    summary = event.get("event_summary") or sample.get("event_summary", "")
    return f"{title} {summary}".strip()


def build_company_text(company):
    if not company:
        return ""
    parts = [
        company.get("stock_name", ""),
        company.get("summary_for_matching", ""),
        company.get("business_desc", ""),
    ]
    return " ".join(part for part in parts if part).strip()


def build_event_num(event, category_map):
    base = [
        safe_float(event.get("heat")),
        safe_float(event.get("event_intensity")),
        safe_float(event.get("influence_range")),
        safe_float(event.get("attribute_score")),
    ]
    return (
        base
        + one_hot(event.get("event_type", "未知"), category_map["event_types"])
        + one_hot(event.get("duration_type", "未知"), category_map["duration_types"])
    )


def build_company_num(company, assoc_score, category_map):
    if not company:
        return [0.0] * (len(category_map["industry_types"]) + len(category_map["chain_positions"]) + 1)

    industry_vec = one_hot(company.get("industry_lv1", "未知"), category_map["industry_types"])
    chain_vec = multi_hot(company.get("industry_chain_position", []) or ["未知"], category_map["chain_positions"])
    return industry_vec + chain_vec + [safe_float(assoc_score)]


def extract_price_vector(day_row):
    row = normalize_stock_row(day_row)
    return [row[field] for field in PRICE_FIELDS]


def compute_delta(window_rows):
    first = np.array(extract_price_vector(window_rows[0]), dtype=np.float32)
    second = np.array(extract_price_vector(window_rows[1]), dtype=np.float32)
    return (second - first).tolist()


def compute_car4(sample, stock_returns, benchmark_returns):
    stock_code = normalize_stock_code(sample["stock_code"])
    future_rows = sample.get("future_window_post4", [])
    if len(future_rows) < 4:
        return None

    stock_ret_lookup = stock_returns.get(stock_code, {})
    car_value = 0.0
    for row in future_rows[:4]:
        date = str(row["date"])
        stock_ret = stock_ret_lookup.get(date)
        if stock_ret is None:
            pct_change = safe_float(row.get("pct_change"))
            stock_ret = pct_change / 100.0
        benchmark_ret = benchmark_returns.get(date)
        if benchmark_ret is None:
            return None
        car_value += stock_ret - benchmark_ret
    return float(car_value)


def enrich_samples(split_name, samples, event_lookup, company_lookup, stock_returns, benchmark_returns, category_map):
    enriched = []
    dropped = []

    for sample in samples:
        event_id = sample["event_id"]
        event = event_lookup.get(event_id)
        if event is None:
            dropped.append({"sample_id": sample.get("sample_id"), "reason": "missing_event"})
            continue

        stock_code = normalize_stock_code(sample["stock_code"])
        company = company_lookup.get(stock_code)

        window = sample.get("input_window_pre2", [])
        if len(window) != 2:
            dropped.append({"sample_id": sample.get("sample_id"), "reason": "invalid_window"})
            continue

        time_seq = [extract_price_vector(window[0]), extract_price_vector(window[1])]
        delta_feat = compute_delta(window)

        car4 = compute_car4(sample, stock_returns, benchmark_returns)
        cls_label = int(car4 > 0) if car4 is not None else None

        enriched.append(
            {
                "sample_id": sample["sample_id"],
                "event_id": event_id,
                "stock_code": stock_code,
                "event_date": sample["event_date"],
                "label_start_date": sample.get("label_start_date"),
                "split": split_name,
                "event_text": build_event_text(sample, event),
                "event_num": build_event_num(event, category_map),
                "company_text": build_company_text(company),
                "company_num": build_company_num(company, sample.get("relation_score", 0.0), category_map),
                "time_seq": time_seq,
                "delta_feat": delta_feat,
                "assoc_score": safe_float(sample.get("relation_score", 0.0)),
                "car4": car4,
                "cls_label": cls_label,
                "has_label": car4 is not None,
            }
        )

    return enriched, dropped


def fit_text_vectorizer(texts, max_features):
    return CharTfidfVectorizer(max_features=max_features, ngram_range=(1, 2)).fit(texts)


def fit_standardizer(matrix):
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    return {"mean": mean, "std": std}


def apply_standardizer(matrix, scaler):
    return (matrix - scaler["mean"]) / scaler["std"]


def samples_to_arrays(samples, event_vectorizer, company_vectorizer, scalers):
    event_text = event_vectorizer.transform([sample["event_text"] for sample in samples]).astype(np.float32)
    company_text = company_vectorizer.transform([sample["company_text"] for sample in samples]).astype(np.float32)

    event_num = np.array([sample["event_num"] for sample in samples], dtype=np.float32)
    company_num = np.array([sample["company_num"] for sample in samples], dtype=np.float32)
    time_seq = np.array([sample["time_seq"] for sample in samples], dtype=np.float32)
    delta_feat = np.array([sample["delta_feat"] for sample in samples], dtype=np.float32)

    event_num = apply_standardizer(event_num, scalers["event_num"]).astype(np.float32)
    company_num = apply_standardizer(company_num, scalers["company_num"]).astype(np.float32)
    flat_time = time_seq.reshape(-1, time_seq.shape[-1])
    flat_time = apply_standardizer(flat_time, scalers["time_seq"]).astype(np.float32)
    time_seq = flat_time.reshape(time_seq.shape).astype(np.float32)
    delta_feat = apply_standardizer(delta_feat, scalers["delta_feat"]).astype(np.float32)

    cls_label = np.array(
        [-1.0 if sample["cls_label"] is None else float(sample["cls_label"]) for sample in samples],
        dtype=np.float32,
    )
    reg_label = np.array(
        [0.0 if sample["car4"] is None else float(sample["car4"]) for sample in samples],
        dtype=np.float32,
    )

    return {
        "event_text": event_text,
        "event_num": event_num,
        "company_text": company_text,
        "company_num": company_num,
        "time_seq": time_seq,
        "delta_feat": delta_feat,
        "cls_label": cls_label,
        "reg_label": reg_label,
        "meta": samples,
    }


class SampleDataset(Dataset):
    def __init__(self, arrays):
        self.arrays = arrays

    def __len__(self):
        return len(self.arrays["meta"])

    def __getitem__(self, idx):
        return (
            torch.tensor(self.arrays["event_text"][idx], dtype=torch.float32),
            torch.tensor(self.arrays["event_num"][idx], dtype=torch.float32),
            torch.tensor(self.arrays["company_text"][idx], dtype=torch.float32),
            torch.tensor(self.arrays["company_num"][idx], dtype=torch.float32),
            torch.tensor(self.arrays["time_seq"][idx], dtype=torch.float32),
            torch.tensor(self.arrays["delta_feat"][idx], dtype=torch.float32),
            torch.tensor(self.arrays["cls_label"][idx], dtype=torch.float32),
            torch.tensor(self.arrays["reg_label"][idx], dtype=torch.float32),
        )


def collate_fn(batch):
    items = list(zip(*batch))
    return tuple(torch.stack(item) for item in items)


def train_one_epoch(model, loader, optimizer, criterion_cls, criterion_reg, device):
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    total_count = 0

    for batch in loader:
        event_text, event_num, company_text, company_num, time_seq, delta_feat, cls_label, reg_label = batch
        event_text = event_text.to(device)
        event_num = event_num.to(device)
        company_text = company_text.to(device)
        company_num = company_num.to(device)
        time_seq = time_seq.to(device)
        delta_feat = delta_feat.to(device)
        cls_label = cls_label.to(device)
        reg_label = reg_label.to(device)

        optimizer.zero_grad()
        outputs = model(event_text, event_num, company_text, company_num, time_seq, delta_feat)

        cls_loss = criterion_cls(outputs["cls_logit"], cls_label)
        reg_loss = criterion_reg(outputs["reg_value"], reg_label)
        loss = ALPHA * cls_loss + BETA * reg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = cls_label.size(0)
        total_loss += loss.item() * batch_size
        total_cls += cls_loss.item() * batch_size
        total_reg += reg_loss.item() * batch_size
        total_count += batch_size

    return {
        "loss": total_loss / total_count,
        "cls_loss": total_cls / total_count,
        "reg_loss": total_reg / total_count,
    }


@torch.no_grad()
def evaluate(model, loader, arrays, criterion_cls, criterion_reg, device):
    model.eval()
    logits = []
    reg_values = []
    cls_labels = []
    reg_labels = []

    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    total_count = 0

    for batch in loader:
        event_text, event_num, company_text, company_num, time_seq, delta_feat, cls_label, reg_label = batch
        event_text = event_text.to(device)
        event_num = event_num.to(device)
        company_text = company_text.to(device)
        company_num = company_num.to(device)
        time_seq = time_seq.to(device)
        delta_feat = delta_feat.to(device)
        cls_label = cls_label.to(device)
        reg_label = reg_label.to(device)

        outputs = model(event_text, event_num, company_text, company_num, time_seq, delta_feat)
        cls_loss = criterion_cls(outputs["cls_logit"], cls_label)
        reg_loss = criterion_reg(outputs["reg_value"], reg_label)
        loss = ALPHA * cls_loss + BETA * reg_loss

        batch_size = cls_label.size(0)
        total_loss += loss.item() * batch_size
        total_cls += cls_loss.item() * batch_size
        total_reg += reg_loss.item() * batch_size
        total_count += batch_size

        logits.append(outputs["cls_logit"].cpu().numpy())
        reg_values.append(outputs["reg_value"].cpu().numpy())
        cls_labels.append(cls_label.cpu().numpy())
        reg_labels.append(reg_label.cpu().numpy())

    logits = np.concatenate(logits) if logits else np.array([], dtype=np.float32)
    reg_values = np.concatenate(reg_values) if reg_values else np.array([], dtype=np.float32)
    cls_labels = np.concatenate(cls_labels) if cls_labels else np.array([], dtype=np.float32)
    reg_labels = np.concatenate(reg_labels) if reg_labels else np.array([], dtype=np.float32)

    metrics = compute_metrics(
        logits=logits,
        reg_values=reg_values,
        cls_labels=cls_labels,
        reg_labels=reg_labels,
        meta=arrays["meta"],
        labeled_only=True,
    )
    metrics["loss"] = total_loss / total_count
    metrics["cls_loss"] = total_cls / total_count
    metrics["reg_loss"] = total_reg / total_count
    return metrics


def binary_classification_metrics(true_cls, pred_cls):
    true_cls = np.asarray(true_cls, dtype=np.int32)
    pred_cls = np.asarray(pred_cls, dtype=np.int32)

    tp = int(np.sum((true_cls == 1) & (pred_cls == 1)))
    tn = int(np.sum((true_cls == 0) & (pred_cls == 0)))
    fp = int(np.sum((true_cls == 0) & (pred_cls == 1)))
    fn = int(np.sum((true_cls == 1) & (pred_cls == 0)))

    total = max(len(true_cls), 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1


def binary_auc_score(true_cls, probs):
    true_cls = np.asarray(true_cls, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float64)
    pos_mask = true_cls == 1
    neg_mask = true_cls == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_probs = probs[order]
    idx = 0
    while idx < len(sorted_probs):
        jdx = idx + 1
        while jdx < len(sorted_probs) and sorted_probs[jdx] == sorted_probs[idx]:
            jdx += 1
        avg_rank = (idx + jdx - 1) / 2.0 + 1.0
        ranks[order[idx:jdx]] = avg_rank
        idx = jdx

    rank_sum_pos = float(np.sum(ranks[pos_mask]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


def compute_metrics(logits, reg_values, cls_labels, reg_labels, meta, labeled_only):
    if labeled_only:
        mask = cls_labels >= 0
    else:
        mask = np.ones_like(cls_labels, dtype=bool)

    logits = logits[mask]
    reg_values = reg_values[mask]
    cls_labels = cls_labels[mask]
    reg_labels = reg_labels[mask]
    filtered_meta = [sample for sample, keep in zip(meta, mask.tolist()) if keep]

    if len(filtered_meta) == 0:
        return {
            "n": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "auc": None,
            "mae": None,
            "rmse": None,
            "r2": None,
            "topk_true_car_mean": None,
        }

    probs = 1.0 / (1.0 + np.exp(-logits))
    pred_cls = (probs >= 0.5).astype(int)
    true_cls = cls_labels.astype(int)
    accuracy, precision, recall, f1 = binary_classification_metrics(true_cls, pred_cls)
    auc = binary_auc_score(true_cls, probs)
    mae = float(np.mean(np.abs(reg_labels - reg_values)))
    rmse = float(math.sqrt(np.mean((reg_values - reg_labels) ** 2)))
    reg_mean = float(np.mean(reg_labels))
    total_var = float(np.sum((reg_labels - reg_mean) ** 2))
    residual_var = float(np.sum((reg_labels - reg_values) ** 2))
    r2 = None if len(filtered_meta) <= 1 or total_var == 0 else float(1.0 - residual_var / total_var)

    metrics = {
        "n": int(len(filtered_meta)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "topk_true_car_mean": None,
    }

    grouped = defaultdict(list)
    for sample, pred_value, true_value in zip(filtered_meta, reg_values.tolist(), reg_labels.tolist()):
        grouped[sample["event_id"]].append((pred_value, true_value))

    per_event_topk_true = []
    for rows in grouped.values():
        rows.sort(key=lambda item: item[0], reverse=True)
        top_rows = rows[: min(TOP_K_EFFECTIVENESS, len(rows))]
        per_event_topk_true.append(float(np.mean([item[1] for item in top_rows])))
    if per_event_topk_true:
        metrics["topk_true_car_mean"] = float(np.mean(per_event_topk_true))

    return metrics


@torch.no_grad()
def predict(model, arrays, device):
    model.eval()
    dataset = SampleDataset(arrays)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    logits = []
    reg_values = []

    for batch in loader:
        event_text, event_num, company_text, company_num, time_seq, delta_feat, _, _ = batch
        outputs = model(
            event_text.to(device),
            event_num.to(device),
            company_text.to(device),
            company_num.to(device),
            time_seq.to(device),
            delta_feat.to(device),
        )
        logits.append(outputs["cls_logit"].cpu().numpy())
        reg_values.append(outputs["reg_value"].cpu().numpy())

    logits = np.concatenate(logits) if logits else np.array([], dtype=np.float32)
    reg_values = np.concatenate(reg_values) if reg_values else np.array([], dtype=np.float32)
    probs = 1.0 / (1.0 + np.exp(-logits))

    rows = []
    for sample, prob, reg_value in zip(arrays["meta"], probs.tolist(), reg_values.tolist()):
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "event_id": sample["event_id"],
                "stock_code": sample["stock_code"],
                "event_date": sample["event_date"],
                "label_start_date": sample["label_start_date"],
                "pred_prob_up": float(prob),
                "pred_car4": float(reg_value),
                "true_cls_label": None if sample["cls_label"] is None else int(sample["cls_label"]),
                "true_car4": sample["car4"],
            }
        )
    return rows


def save_prediction_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_matrix_csv(path: Path, rows, value_key: str):
    event_ids = sorted({row["event_id"] for row in rows})
    stock_codes = sorted({row["stock_code"] for row in rows})
    value_lookup = {(row["stock_code"], row["event_id"]): row[value_key] for row in rows}

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stock_code"] + event_ids)
        for stock_code in stock_codes:
            writer.writerow([stock_code] + [value_lookup.get((stock_code, event_id)) for event_id in event_ids])


def probs_to_logits(rows):
    logits = []
    for row in rows:
        prob = min(max(float(row["pred_prob_up"]), 1e-6), 1.0 - 1e-6)
        logits.append(math.log(prob / (1.0 - prob)))
    return np.array(logits, dtype=np.float32)


def main() -> int:
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Task3] device={device}", flush=True)

    split_payloads = {
        "train": load_json(TRAIN_SAMPLES_PATH)["samples"],
        "val": load_json(VAL_SAMPLES_PATH)["samples"],
        "pending": load_json(PENDING_SAMPLES_PATH)["samples"],
    }
    event_lookup = load_event_lookup()
    company_lookup = load_company_lookup()
    stock_history = load_stock_history(STOCK_HISTORY_PATHS)
    stock_returns, benchmark_returns = compute_daily_returns(stock_history)

    all_base_samples = split_payloads["train"] + split_payloads["val"] + split_payloads["pending"]
    category_map = collect_categories(all_base_samples, event_lookup, company_lookup)

    split_samples = {}
    dropped_records = []
    for split_name, samples in split_payloads.items():
        enriched, dropped = enrich_samples(
            split_name,
            samples,
            event_lookup,
            company_lookup,
            stock_returns,
            benchmark_returns,
            category_map,
        )
        split_samples[split_name] = enriched
        dropped_records.extend({"split": split_name, **item} for item in dropped)

    train_samples = [sample for sample in split_samples["train"] if sample["has_label"]]
    val_samples = [sample for sample in split_samples["val"] if sample["has_label"]]
    pending_samples = split_samples["pending"]

    print(
        f"[Task3] samples train={len(train_samples)} val={len(val_samples)} pending={len(pending_samples)} dropped={len(dropped_records)}",
        flush=True,
    )

    event_vectorizer = fit_text_vectorizer([sample["event_text"] for sample in train_samples], max_features=128)
    company_vectorizer = fit_text_vectorizer([sample["company_text"] for sample in train_samples], max_features=192)

    train_event_num = np.array([sample["event_num"] for sample in train_samples], dtype=np.float32)
    train_company_num = np.array([sample["company_num"] for sample in train_samples], dtype=np.float32)
    train_time_seq = np.array([sample["time_seq"] for sample in train_samples], dtype=np.float32).reshape(-1, len(PRICE_FIELDS))
    train_delta_feat = np.array([sample["delta_feat"] for sample in train_samples], dtype=np.float32)

    scalers = {
        "event_num": fit_standardizer(train_event_num),
        "company_num": fit_standardizer(train_company_num),
        "time_seq": fit_standardizer(train_time_seq),
        "delta_feat": fit_standardizer(train_delta_feat),
    }

    train_arrays = samples_to_arrays(train_samples, event_vectorizer, company_vectorizer, scalers)
    val_arrays = samples_to_arrays(val_samples, event_vectorizer, company_vectorizer, scalers)
    pending_arrays = samples_to_arrays(pending_samples, event_vectorizer, company_vectorizer, scalers)

    artifacts = {
        "category_map": category_map,
        "event_vocabulary": event_vectorizer.vocabulary_,
        "company_vocabulary": company_vectorizer.vocabulary_,
        "scalers": {
            key: {"mean": value["mean"].tolist(), "std": value["std"].tolist()}
            for key, value in scalers.items()
        },
        "benchmark_note": "Used equal-weight daily return of available HS300 constituent histories as market benchmark proxy because official 000300 index history was not found locally.",
    }
    with (OUTPUT_DIR / "artifacts.pkl").open("wb") as f:
        pickle.dump(artifacts, f)

    model = PaperEventLSTM(
        event_text_dim=train_arrays["event_text"].shape[1],
        event_num_dim=train_arrays["event_num"].shape[1],
        company_text_dim=train_arrays["company_text"].shape[1],
        company_num_dim=train_arrays["company_num"].shape[1],
        time_input_dim=len(PRICE_FIELDS),
        delta_dim=len(PRICE_FIELDS),
    ).to(device)
    print(f"[Task3] params={count_params(model):,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.HuberLoss(delta=1.0)

    train_loader = DataLoader(SampleDataset(train_arrays), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(SampleDataset(val_arrays), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    history = []
    best_state = None
    best_val_loss = float("inf")
    wait = 0

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion_cls, criterion_reg, device)
        val_metrics = evaluate(model, val_loader, val_arrays, criterion_cls, criterion_reg, device)

        entry = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_cls_loss": round(train_metrics["cls_loss"], 6),
            "train_reg_loss": round(train_metrics["reg_loss"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_cls_loss": round(val_metrics["cls_loss"], 6),
            "val_reg_loss": round(val_metrics["reg_loss"], 6),
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "val_topk_true_car_mean": val_metrics["topk_true_car_mean"],
        }
        history.append(entry)

        print(
            f"Epoch {epoch:02d} | train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['auc']} "
            f"val_f1={val_metrics['f1']:.4f} val_rmse={val_metrics['rmse']:.4f}",
            flush=True,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"[Task3] early stop at epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)

    checkpoint = {
        "model_state": best_state,
        "model_config": {
            "event_text_dim": train_arrays["event_text"].shape[1],
            "event_num_dim": train_arrays["event_num"].shape[1],
            "company_text_dim": train_arrays["company_text"].shape[1],
            "company_num_dim": train_arrays["company_num"].shape[1],
            "time_input_dim": len(PRICE_FIELDS),
            "delta_dim": len(PRICE_FIELDS),
        },
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, OUTPUT_DIR / "best_model.pt")

    train_predictions = predict(model, train_arrays, device)
    val_predictions = predict(model, val_arrays, device)
    pending_predictions = predict(model, pending_arrays, device)

    save_prediction_csv(OUTPUT_DIR / "train_predictions.csv", train_predictions)
    save_prediction_csv(OUTPUT_DIR / "val_predictions.csv", val_predictions)
    save_prediction_csv(OUTPUT_DIR / "pending_predictions.csv", pending_predictions)
    write_json(OUTPUT_DIR / "train_predictions.json", train_predictions)
    write_json(OUTPUT_DIR / "val_predictions.json", val_predictions)
    write_json(OUTPUT_DIR / "pending_predictions.json", pending_predictions)

    save_matrix_csv(OUTPUT_DIR / "pending_probability_matrix.csv", pending_predictions, "pred_prob_up")
    save_matrix_csv(OUTPUT_DIR / "pending_regression_matrix.csv", pending_predictions, "pred_car4")

    final_train_metrics = compute_metrics(
        logits=probs_to_logits(train_predictions),
        reg_values=np.array([row["pred_car4"] for row in train_predictions], dtype=np.float32),
        cls_labels=np.array([row["true_cls_label"] for row in train_predictions], dtype=np.float32),
        reg_labels=np.array([row["true_car4"] for row in train_predictions], dtype=np.float32),
        meta=train_samples,
        labeled_only=True,
    )
    final_val_metrics = compute_metrics(
        logits=probs_to_logits(val_predictions),
        reg_values=np.array([row["pred_car4"] for row in val_predictions], dtype=np.float32),
        cls_labels=np.array([row["true_cls_label"] for row in val_predictions], dtype=np.float32),
        reg_labels=np.array([row["true_car4"] for row in val_predictions], dtype=np.float32),
        meta=val_samples,
        labeled_only=True,
    )

    summary = {
        "assumption": "Used equal-weight return of available HS300 constituent histories as benchmark proxy for abnormal return.",
        "split_sizes": {
            "train": len(train_samples),
            "val": len(val_samples),
            "pending": len(pending_samples),
            "dropped": len(dropped_records),
        },
        "feature_dims": {
            "event_text_dim": train_arrays["event_text"].shape[1],
            "event_num_dim": train_arrays["event_num"].shape[1],
            "company_text_dim": train_arrays["company_text"].shape[1],
            "company_num_dim": train_arrays["company_num"].shape[1],
            "time_seq_dim": len(PRICE_FIELDS),
            "delta_dim": len(PRICE_FIELDS),
        },
        "best_val_loss": best_val_loss,
        "train_metrics": final_train_metrics,
        "val_metrics": final_val_metrics,
        "pending_top10_by_regression": sorted(
            pending_predictions,
            key=lambda row: row["pred_car4"],
            reverse=True,
        )[:10],
    }

    write_json(OUTPUT_DIR / "training_history.json", history)
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_json(OUTPUT_DIR / "dropped_records.json", dropped_records)
    write_json(
        OUTPUT_DIR / "enriched_dataset.json",
        {
            "train": train_samples,
            "val": val_samples,
            "pending": pending_samples,
        },
    )

    print()
    print("=" * 72)
    print("[Task3] training finished")
    print(f"  output_dir: {OUTPUT_DIR}")
    print(f"  best_val_loss: {best_val_loss:.6f}")
    print(f"  val_auc: {final_val_metrics['auc']}")
    print(f"  val_f1: {final_val_metrics['f1']}")
    print(f"  val_rmse: {final_val_metrics['rmse']}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
