import json
from pathlib import Path


SOURCE_PATH = Path(r"D:\Math_match\codes\outputs\step1_detection_semisupervised_eastmoney400.json")
TARGET_PATH = Path(r"D:\Math_match\codes\outputs\step1_event_only_for_classification_eastmoney400.json")


def main() -> int:
    payload = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    grouped_results = payload.get("grouped_results", {})

    filtered_grouped = {}
    event_count = 0
    site_event_counts = {}
    bucket_event_counts = {
        "high_confidence_positive": 0,
        "high_confidence_negative": 0,
        "medium_confidence_positive": 0,
        "medium_confidence_negative": 0,
        "boundary_positive": 0,
        "boundary_negative": 0,
    }

    for source_site, items in grouped_results.items():
        if not isinstance(items, list):
            continue
        filtered_items = [item for item in items if int(item.get("is_event", 0)) == 1]
        if not filtered_items:
            continue
        filtered_grouped[source_site] = filtered_items
        site_event_counts[source_site] = len(filtered_items)
        event_count += len(filtered_items)
        for item in filtered_items:
            bucket = str(item.get("seed_category") or item.get("sample_bucket") or "")
            if bucket in bucket_event_counts:
                bucket_event_counts[bucket] += 1

    target_payload = {
        "source_file": str(SOURCE_PATH),
        "run_status": payload.get("run_status", ""),
        "progress": payload.get("progress", {}),
        "seed_count": payload.get("seed_count", 0),
        "event_count": event_count,
        "site_event_counts": site_event_counts,
        "bucket_event_counts": bucket_event_counts,
        "grouped_results": filtered_grouped,
    }
    TARGET_PATH.write_text(json.dumps(target_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Event-only file refreshed: {TARGET_PATH}")
    print(f"Event count: {event_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
