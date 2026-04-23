import os
import sys
from pathlib import Path

from task2.semantic_match import SemanticMatchConfig, run_semantic_matching


# Task 2 stage-2 semantic matching configuration
COMPANY_PROFILES_PATH = r"D:\Math_match\codes\outputs\task2_profile_run\company_profiles.json"
EVENTS_PATH = r"D:\Math_match\codes\outputs\classification_run_eastmoney400_v2\classification_results.json"
OUTPUT_DIR = r"D:\Math_match\codes\outputs\task2_semantic_run"

MODEL_NAME = "BAAI/bge-large-zh-v1.5"
TOP_K = 20
BATCH_SIZE = 32
DEVICE = ""  # "" for auto; set "cpu" or "cuda" if needed.


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    cfg = SemanticMatchConfig(
        model_name=MODEL_NAME,
        top_k=TOP_K,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    result = run_semantic_matching(
        company_profiles_path=COMPANY_PROFILES_PATH,
        events_path=EVENTS_PATH,
        output_dir=OUTPUT_DIR,
        config=cfg,
    )

    meta = result.get("meta", {})
    print(f"[Task2-S0] 模型: {meta.get('model_name', MODEL_NAME)}")
    print(f"[Task2-S0] 事件数: {meta.get('event_count', 0)} | 公司数: {meta.get('company_count', 0)}")
    print(f"[Task2-S0] 输出目录: {Path(OUTPUT_DIR).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
