import json
from pathlib import Path
from typing import Dict, Iterable, List

ANNOTATIONS_PATH = Path(r"D:\Math_match\codes\outputs\step1_detection_annotations_eastmoney400.json")
OUTPUT_PATH = Path(r"D:\Math_match\codes\outputs\step1_detection_seed_12_eastmoney400.json")

CORE_POSITIVE_COUNT = 6
CORE_NEGATIVE_COUNT = 4
AUX_POSITIVE_COUNT = 2
BOUNDARY_COUNT = 0


def _sorted_bucket(items: Iterable[dict], reverse: bool) -> List[dict]:
    def key(item: dict):
        confidence = float(item.get("final_confidence", 0.0) or 0.0)
        sample_id = str(item.get("sample_id") or "")
        return (confidence, sample_id)

    return sorted(list(items), key=key, reverse=reverse)


def _clone(items: Iterable[dict]) -> List[dict]:
    keys = [
        "sample_id",
        "source_site",
        "title",
        "url",
        "is_event",
        "has_new_fact",
        "has_market_impact_path",
        "reason",
        "final_confidence",
        "seed_category",
    ]
    result = []
    for item in items:
        result.append({key: item.get(key) for key in keys})
    return result


def _ensure_enough(name: str, actual: int, required: int) -> None:
    if actual < required:
        raise ValueError(f"{name} 数量不足：需要 {required} 条，实际只有 {actual} 条。请先检查 annotations 结果。")


def main() -> int:
    payload = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    bucketed: Dict[str, List[dict]] = payload.get("bucketed_results", {})

    high_pos = _sorted_bucket(bucketed.get("high_confidence_positive", []), reverse=True)
    high_neg = _sorted_bucket(bucketed.get("high_confidence_negative", []), reverse=False)
    medium_pos = _sorted_bucket(bucketed.get("medium_confidence_positive", []), reverse=True)
    boundary_pos = _sorted_bucket(bucketed.get("boundary_positive", []), reverse=True)

    _ensure_enough("high_confidence_positive", len(high_pos), CORE_POSITIVE_COUNT)
    _ensure_enough("high_confidence_negative", len(high_neg), CORE_NEGATIVE_COUNT)
    _ensure_enough("medium_confidence_positive", len(medium_pos), AUX_POSITIVE_COUNT)

    core_positive = high_pos[:CORE_POSITIVE_COUNT]
    core_negative = high_neg[:CORE_NEGATIVE_COUNT]

    used_ids = {str(item.get("sample_id") or "") for item in core_positive + core_negative}
    auxiliary_positive = [
        item for item in medium_pos
        if str(item.get("sample_id") or "") not in used_ids
    ][:AUX_POSITIVE_COUNT]
    _ensure_enough("auxiliary_seed_samples", len(auxiliary_positive), AUX_POSITIVE_COUNT)

    used_ids.update(str(item.get("sample_id") or "") for item in auxiliary_positive)
    boundary_samples = [
        item for item in boundary_pos
        if str(item.get("sample_id") or "") not in used_ids
    ][:BOUNDARY_COUNT]

    seed_payload = {
        "selection_strategy": {
            "core_seed_samples": "6 high-confidence positives + 4 high-confidence negatives",
            "auxiliary_seed_samples": "2 medium-confidence positives",
            "boundary_seed_samples": "0 for current round",
            "usage_note": "core seeds are the main supervision signals; auxiliary seeds are few-shot support only.",
            "source_annotations": str(ANNOTATIONS_PATH),
        },
        "core_seed_samples": _clone(core_positive + core_negative),
        "auxiliary_seed_samples": _clone(auxiliary_positive),
        "boundary_seed_samples": _clone(boundary_samples),
    }

    OUTPUT_PATH.write_text(json.dumps(seed_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Seed file written: {OUTPUT_PATH}")
    print(f"core_seed_samples: {len(seed_payload['core_seed_samples'])}")
    print(f"auxiliary_seed_samples: {len(seed_payload['auxiliary_seed_samples'])}")
    print(f"boundary_seed_samples: {len(seed_payload['boundary_seed_samples'])}")
    print(f"available high_confidence_positive: {len(high_pos)}")
    print(f"available high_confidence_negative: {len(high_neg)}")
    print(f"available medium_confidence_positive: {len(medium_pos)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
