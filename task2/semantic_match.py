import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class SemanticMatchConfig:
    model_name: str = "BAAI/bge-large-zh-v1.5"
    top_k: int = 20
    batch_size: int = 32
    device: str = ""


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _join_non_empty(parts: Iterable[str], sep: str = "；") -> str:
    cleaned = [str(part).strip() for part in parts if str(part).strip()]
    return sep.join(cleaned)


def _profile_text(profile: Dict[str, Any]) -> str:
    fields = [
        profile.get("stock_name", ""),
        profile.get("company_full_name", ""),
        profile.get("industry_lv1", ""),
        profile.get("industry_lv2", ""),
        profile.get("business_desc", ""),
        profile.get("summary_for_matching", ""),
    ]
    list_fields = [
        "concept_tags",
        "main_products",
        "product_keywords",
        "application_scenarios",
        "industry_chain_position",
        "upstream_keywords",
        "downstream_keywords",
        "event_sensitive_keywords",
        "relation_entities",
        "direct_match_aliases",
        "industry_nodes",
        "event_match_keywords",
    ]
    for key in list_fields:
        fields.extend(_ensure_list(profile.get(key)))
    return _join_non_empty(fields)


def _event_text(event: Dict[str, Any]) -> str:
    summary = str(event.get("event_summary") or "").strip()
    if not summary:
        summary = _join_non_empty(
            [
                event.get("title", ""),
                event.get("event_type", ""),
                event.get("event_attribute", ""),
                event.get("classification_reason", ""),
                event.get("correlation_logic", ""),
            ]
        )
    return summary


def _load_company_profiles(path: Path) -> Tuple[List[Dict[str, Any]], List[str], List[str], List[str]]:
    payload = _load_json(path)
    items = payload.get("companies", []) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("company_profiles input must be a JSON array or {'companies': [...]}.")

    records: List[Dict[str, Any]] = []
    codes: List[str] = []
    names: List[str] = []
    texts: List[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        stock_code = str(item.get("stock_code") or "").strip()
        stock_name = str(item.get("stock_name") or "").strip()
        if not stock_code or not stock_name:
            continue
        text = _profile_text(item)
        if not text:
            text = _join_non_empty([stock_name, item.get("industry_lv1", "")])
        records.append(item)
        codes.append(stock_code)
        names.append(stock_name)
        texts.append(text)

    if not records:
        raise ValueError(f"No valid company profiles found in {path}")
    return records, codes, names, texts


def _load_events(path: Path) -> Tuple[List[Dict[str, Any]], List[str], List[str], List[str], List[str], List[float]]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError("events input must be a JSON array.")

    records: List[Dict[str, Any]] = []
    sample_ids: List[str] = []
    titles: List[str] = []
    types: List[str] = []
    texts: List[str] = []
    attributes: List[float] = []

    for item in payload:
        if not isinstance(item, dict):
            continue
        sample_id = str(item.get("sample_id") or item.get("document_id") or "").strip()
        title = str(item.get("title") or "").strip()
        event_type = str(item.get("event_type") or "").strip()
        if not sample_id:
            continue
        text = _event_text(item)
        if not text:
            continue
        attr = float(item.get("attribute_score") or 0.0)
        records.append(item)
        sample_ids.append(sample_id)
        titles.append(title)
        types.append(event_type)
        texts.append(text)
        attributes.append(attr)

    if not records:
        raise ValueError(f"No valid events found in {path}")
    return records, sample_ids, titles, types, texts, attributes


def _load_embedding_model(model_name: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for semantic matching. "
            "Please install: pip install sentence-transformers"
        ) from exc

    kwargs = {}
    if device and device != "auto":
        kwargs["device"] = device
    return SentenceTransformer(model_name, **kwargs)


def _encode_texts(model, texts: List[str], batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings)
    return embeddings.astype(np.float32)


def run_semantic_matching(
    company_profiles_path: str,
    events_path: str,
    output_dir: str,
    config: SemanticMatchConfig,
) -> Dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    profile_records, company_codes, company_names, profile_texts = _load_company_profiles(Path(company_profiles_path))
    event_records, event_ids, event_titles, event_types, event_texts, event_attributes = _load_events(Path(events_path))

    print(f"[Task2-S0] 加载公司画像 {len(company_codes)} 家，事件 {len(event_ids)} 条。", flush=True)
    print(f"[Task2-S0] 正在加载语义模型: {config.model_name}", flush=True)
    model = _load_embedding_model(config.model_name, config.device)

    print("[Task2-S0] 编码公司画像文本...", flush=True)
    company_emb = _encode_texts(model, profile_texts, config.batch_size)
    print("[Task2-S0] 编码事件摘要文本...", flush=True)
    event_emb = _encode_texts(model, event_texts, config.batch_size)

    # Both embeddings are normalized, so dot product equals cosine similarity.
    cosine_matrix = np.matmul(event_emb, company_emb.T)
    s0_matrix = np.clip((1.0 + cosine_matrix) / 2.0, 0.0, 1.0)

    np.save(output / "s0_matrix.npy", s0_matrix)
    np.save(output / "cosine_matrix.npy", cosine_matrix)

    matrix_df = pd.DataFrame(s0_matrix, index=event_ids, columns=company_codes)
    matrix_df.to_csv(output / "s0_matrix.csv", encoding="utf-8-sig")

    pair_rows: List[Dict[str, Any]] = []
    topk_rows: List[Dict[str, Any]] = []

    for event_idx, sample_id in enumerate(event_ids):
        scores = s0_matrix[event_idx]
        cos_scores = cosine_matrix[event_idx]
        order = np.argsort(scores)[::-1]
        k = min(config.top_k, len(order))

        top_companies: List[Dict[str, Any]] = []
        for rank in range(k):
            comp_idx = int(order[rank])
            top_companies.append(
                {
                    "rank": rank + 1,
                    "stock_code": company_codes[comp_idx],
                    "stock_name": company_names[comp_idx],
                    "cosine_similarity": float(round(cos_scores[comp_idx], 6)),
                    "s0": float(round(scores[comp_idx], 6)),
                }
            )

        topk_rows.append(
            {
                "sample_id": sample_id,
                "title": event_titles[event_idx],
                "event_type": event_types[event_idx],
                "event_summary": event_texts[event_idx],
                "attribute_score": float(event_attributes[event_idx]),
                "top_companies": top_companies,
            }
        )

        for comp_idx, stock_code in enumerate(company_codes):
            pair_rows.append(
                {
                    "sample_id": sample_id,
                    "title": event_titles[event_idx],
                    "event_type": event_types[event_idx],
                    "event_summary": event_texts[event_idx],
                    "attribute_score": float(event_attributes[event_idx]),
                    "stock_code": stock_code,
                    "stock_name": company_names[comp_idx],
                    "cosine_similarity": float(round(cos_scores[comp_idx], 6)),
                    "s0": float(round(scores[comp_idx], 6)),
                }
            )

    pd.DataFrame(pair_rows).to_csv(output / "s0_pairs.csv", index=False, encoding="utf-8-sig")

    with (output / "s0_topk.jsonl").open("w", encoding="utf-8") as fh:
        for row in topk_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    (output / "s0_topk.json").write_text(
        json.dumps(topk_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    meta = {
        "model_name": config.model_name,
        "company_profiles_path": str(Path(company_profiles_path).resolve()),
        "events_path": str(Path(events_path).resolve()),
        "company_count": len(company_codes),
        "event_count": len(event_ids),
        "matrix_shape": [int(s0_matrix.shape[0]), int(s0_matrix.shape[1])],
        "matrix_axis": "rows=events, cols=companies",
        "top_k": int(config.top_k),
        "batch_size": int(config.batch_size),
        "device": config.device or "auto",
        "outputs": {
            "s0_pairs_csv": str((output / "s0_pairs.csv").resolve()),
            "s0_topk_jsonl": str((output / "s0_topk.jsonl").resolve()),
            "s0_matrix_npy": str((output / "s0_matrix.npy").resolve()),
            "s0_matrix_csv": str((output / "s0_matrix.csv").resolve()),
        },
    }
    (output / "s0_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "meta": meta,
        "event_ids": event_ids,
        "company_codes": company_codes,
    }
