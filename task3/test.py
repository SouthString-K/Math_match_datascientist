import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import PaperEventLSTM
from trainer import (
    BATCH_SIZE,
    COMPANY_PROFILES_PATH,
    EVENTS_PATH,
    STOCK_HISTORY_PATHS,
    SampleDataset,
    collate_fn,
    collect_categories,
    compute_daily_returns,
    evaluate,
    enrich_samples,
    fit_standardizer,
    fit_text_vectorizer,
    load_company_lookup,
    load_event_lookup,
    load_stock_history,
    predict,
    samples_to_arrays,
    write_json,
)

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_MODEL_PATH = Path(r"D:\Math_match\codes\outputs\task3\paper_training_lstm\best_model.pt")
DEFAULT_DATASET_PATH = Path(r"D:\Math_match\codes\outputs\task3\paper_training_lstm\enriched_dataset.json")
DEFAULT_ARTIFACTS_PATH = Path(r"D:\Math_match\codes\outputs\task3\paper_training_lstm\artifacts.pkl")
DEFAULT_OUTPUT_DIR = Path(r"D:\Math_match\codes\outputs\task3\paper_training_lstm")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_samples(path: Path):
    payload = load_json(path)
    if isinstance(payload, dict):
        if "samples" in payload and isinstance(payload["samples"], list):
            return payload["samples"]
        if all(isinstance(v, list) for v in payload.values()):
            return payload
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported sample file format: {path}")


def is_raw_test_v2(samples):
    if not samples:
        return False
    sample = samples[0]
    return "input_sequence_2d" in sample and "event_trade_date_tau" in sample


def is_raw_test_custom_414(samples):
    if not samples:
        return False
    sample = samples[0]
    return "price_sequence_pre2" in sample and "delta_pre2" in sample and "future_window_post4" in sample


def convert_raw_test_v2_samples(raw_samples):
    converted = []
    for sample in raw_samples:
        converted.append(
            {
                "sample_id": sample["sample_id"],
                "event_date": sample["event_date"],
                "event_trade_date": sample.get("event_trade_date_tau"),
                "label_start_date": sample.get("label_start_date_eta"),
                "stock_code": sample["stock_code"],
                "event_id": sample["event_id"],
                "event_title": sample.get("event_title", ""),
                "event_summary": sample.get("event_summary", ""),
                "relation_score": sample.get("relation_score", 0.0),
                "input_window_pre2": sample.get("input_sequence_2d", []),
                "delta_features": sample.get("delta_features", {}),
                "future_window_post4": sample.get("future_window_4d", []),
                "targets": sample.get("targets", {}),
            }
        )
    return converted


def convert_raw_test_custom_414_samples(raw_samples):
    converted = []
    for sample in raw_samples:
        converted.append(
            {
                "sample_id": sample["sample_id"],
                "event_date": sample["event_date"],
                "event_trade_date": sample.get("event_trade_date"),
                "label_start_date": None,
                "stock_code": sample["stock_code"],
                "event_id": sample["event_id"],
                "event_title": sample.get("event_title", ""),
                "event_summary": sample.get("event_summary", ""),
                "relation_score": sample.get("relation_score", 0.0),
                "input_window_pre2": sample.get("price_sequence_pre2", []),
                "delta_features": sample.get("delta_pre2", {}),
                "future_window_post4": sample.get("future_window_post4", []),
                "targets": sample.get("targets", {}),
                "label_rule": sample.get("label_rule"),
            }
        )
    return converted


def fit_preprocessors(train_samples, model_config):
    event_vectorizer = fit_text_vectorizer(
        [sample["event_text"] for sample in train_samples],
        max_features=model_config["event_text_dim"],
    )
    company_vectorizer = fit_text_vectorizer(
        [sample["company_text"] for sample in train_samples],
        max_features=model_config["company_text_dim"],
    )

    import numpy as np

    train_event_num = np.array([sample["event_num"] for sample in train_samples], dtype=np.float32)
    train_company_num = np.array([sample["company_num"] for sample in train_samples], dtype=np.float32)
    train_time_seq = np.array([sample["time_seq"] for sample in train_samples], dtype=np.float32).reshape(
        -1, model_config["time_input_dim"]
    )
    train_delta_feat = np.array([sample["delta_feat"] for sample in train_samples], dtype=np.float32)

    scalers = {
        "event_num": fit_standardizer(train_event_num),
        "company_num": fit_standardizer(train_company_num),
        "time_seq": fit_standardizer(train_time_seq),
        "delta_feat": fit_standardizer(train_delta_feat),
    }
    return event_vectorizer, company_vectorizer, scalers


def resolve_target_samples(dataset_payload, split: str, custom_samples_path: Optional[Path]):
    if custom_samples_path is not None:
        samples = load_samples(custom_samples_path)
        if isinstance(samples, dict):
            raise ValueError("Custom sample path must resolve to a flat sample list, not a split dict.")
        return samples, custom_samples_path.stem, custom_samples_path

    if not isinstance(dataset_payload, dict):
        raise ValueError("Default enriched dataset must be a split dict with keys like train/val/pending/test.")
    if split not in dataset_payload:
        available = ", ".join(dataset_payload.keys())
        raise ValueError(f"Split '{split}' not found. Available splits: {available}")
    return dataset_payload[split], split, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate task3 best model on a labeled split or custom test set.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to best_model.pt")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to enriched_dataset.json used to fit preprocessors from train split",
    )
    parser.add_argument(
        "--artifacts-path",
        type=Path,
        default=DEFAULT_ARTIFACTS_PATH,
        help="Path to artifacts.pkl from training; used to reuse category_map",
    )
    parser.add_argument("--split", default="val", help="Split name inside enriched_dataset.json, e.g. val/test/pending")
    parser.add_argument(
        "--samples-path",
        type=Path,
        default=None,
        help="Optional custom sample file in enriched sample format; if set, ignores --split for target samples",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save metrics and predictions",
    )
    args = parser.parse_args()

    try:
        checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.model_path, map_location="cpu")
    model_config = checkpoint["model_config"]

    dataset_payload = load_samples(args.dataset_path)
    if not isinstance(dataset_payload, dict) or "train" not in dataset_payload:
        raise ValueError("dataset-path must point to enriched_dataset.json containing at least a 'train' split.")

    train_samples = dataset_payload["train"]
    target_samples, target_name, target_path = resolve_target_samples(dataset_payload, args.split, args.samples_path)

    with args.artifacts_path.open("rb") as f:
        artifacts = pickle.load(f)
    category_map = artifacts["category_map"]

    if target_path is not None and (is_raw_test_v2(target_samples) or is_raw_test_custom_414(target_samples)):
        if is_raw_test_v2(target_samples):
            raw_samples = convert_raw_test_v2_samples(target_samples)
        else:
            raw_samples = convert_raw_test_custom_414_samples(target_samples)
        event_lookup = load_event_lookup()
        company_lookup = load_company_lookup()
        stock_history = load_stock_history(STOCK_HISTORY_PATHS)
        stock_returns, benchmark_returns = compute_daily_returns(stock_history)
        target_samples, dropped = enrich_samples(
            split_name=target_name,
            samples=raw_samples,
            event_lookup=event_lookup,
            company_lookup=company_lookup,
            stock_returns=stock_returns,
            benchmark_returns=benchmark_returns,
            category_map=category_map,
        )
        if dropped:
            print(f"[test] dropped during raw conversion: {len(dropped)}", flush=True)

    event_vectorizer, company_vectorizer, scalers = fit_preprocessors(train_samples, model_config)
    target_arrays = samples_to_arrays(target_samples, event_vectorizer, company_vectorizer, scalers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PaperEventLSTM(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])

    dataset = SampleDataset(target_arrays)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.HuberLoss(delta=1.0)

    metrics = evaluate(model, loader, target_arrays, criterion_cls, criterion_reg, device)
    predictions = predict(model, target_arrays, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{target_name}_metrics.json"
    predictions_path = args.output_dir / f"{target_name}_predictions_from_test.json"
    predictions_csv_path = args.output_dir / f"{target_name}_predictions_from_test.csv"

    write_json(metrics_path, metrics)
    write_json(predictions_path, predictions)

    if predictions:
        import csv

        with predictions_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(predictions[0].keys()))
            writer.writeheader()
            writer.writerows(predictions)

    print(f"[test] device={device}")
    print(f"[test] target={target_name} samples={len(target_samples)}")
    print(f"[test] metrics_path={metrics_path}")
    print(f"[test] predictions_path={predictions_csv_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
