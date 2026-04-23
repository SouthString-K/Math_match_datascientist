import sys
from pathlib import Path

import pandas as pd


LAMBDA = 0.5
S0_PAIRS_PATH = r"D:\Math_match\codes\outputs\task2_semantic_run\s0_pairs.csv"
EVENTS_PATH = r"D:\Math_match\codes\outputs\classification_run_eastmoney400_v2\classification_results.json"
OUTPUT_DIR = r"D:\Math_match\codes\outputs\task2_attribute_correction"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)

    # Step 1: load s0 pairs
    s0_df = pd.read_csv(S0_PAIRS_PATH, encoding="utf-8-sig")
    print(f"[Assoc] 加载 S0 pairs: {len(s0_df)} 条", flush=True)

    # Step 2: load events, compute min-max normalized attribute
    import json
    events_raw = json.loads(Path(EVENTS_PATH).read_text(encoding="utf-8"))
    event_attrs = {
        str(item.get("sample_id") or "").strip(): float(item.get("attribute_score") or 0.0)
        for item in events_raw
        if isinstance(item, dict)
    }

    attr_min = min(event_attrs.values())
    attr_max = max(event_attrs.values())
    print(f"[Assoc] Attribute 范围: [{attr_min:.4f}, {attr_max:.4f}]", flush=True)

    def normalize_attribute(attr: float) -> float:
        if attr_max == attr_min:
            return 0.0
        return (attr - attr_min) / (attr_max - attr_min)

    # Step 3: map attribute_hat to each pair
    attr_hat_series = s0_df["sample_id"].map(
        lambda sid: normalize_attribute(event_attrs.get(str(sid).strip(), 0.0))
    )

    # Step 4: compute Assoc_ij = S0_ij * (1 + lambda * Attribute_hat_j)
    s0_series = s0_df["s0"]
    assoc_series = s0_series * (1.0 + LAMBDA * attr_hat_series)

    assoc_df = s0_df.copy()
    assoc_df["attribute_score"] = s0_df["sample_id"].map(event_attrs)
    assoc_df["attribute_hat"] = attr_hat_series.round(6)
    assoc_df["lambda"] = LAMBDA
    assoc_df["assoc_ij"] = assoc_series.round(6)

    # Step 5: output
    assoc_df.to_csv(output / "assoc_pairs.csv", index=False, encoding="utf-8-sig")
    print(f"[Assoc] 已保存 assoc_pairs.csv: {len(assoc_df)} 条", flush=True)

    # Step 6: pivot matrix (rows=companies, cols=events)
    pivot = assoc_df.pivot_table(
        index="stock_code",
        columns="sample_id",
        values="assoc_ij",
        aggfunc="first",
    )
    pivot.to_csv(output / "assoc_matrix.csv", encoding="utf-8-sig")
    print(f"[Assoc] 已保存 assoc_matrix.csv: {pivot.shape[0]} 家公司 × {pivot.shape[1]} 事件", flush=True)

    summary = {
        "lambda": LAMBDA,
        "s0_pairs_path": str(Path(S0_PAIRS_PATH).resolve()),
        "events_path": str(Path(EVENTS_PATH).resolve()),
        "pair_count": len(assoc_df),
        "company_count": int(pivot.shape[0]),
        "event_count": int(pivot.shape[1]),
        "attribute_min": attr_min,
        "attribute_max": attr_max,
        "formula": "Assoc_ij = S0_ij * (1 + lambda * Attribute_hat_j)",
    }
    (output / "correction_meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[Assoc] 完成。输出目录: {output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())