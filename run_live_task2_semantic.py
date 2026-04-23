"""
实测 Task2：语义匹配（BGE embeddings）
- 复用已有的 300 家公司画像
- 输入 Task1 的分类结果
- 用 BAAI/bge-large-zh-v1.5 计算事件-公司关联矩阵
- 输出 s0_matrix / s0_topk / s0_pairs
"""
import os
import sys
from pathlib import Path

from task2.semantic_match import SemanticMatchConfig, run_semantic_matching

# ---------- 配置 ----------
COMPANY_PROFILES_PATH = r"D:\Math_match\codes\outputs\task2_profile_run\company_profiles.json"
EVENTS_PATH = r"D:\Math_match\codes\outputs\live_classification_eastmoney30\classification_results.json"
OUTPUT_DIR = r"D:\Math_match\codes\outputs\live_task2_semantic"

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
    print(f"[Live-Task2] 模型: {meta.get('model_name', MODEL_NAME)}")
    print(f"[Live-Task2] 事件数: {meta.get('event_count', 0)} | 公司数: {meta.get('company_count', 0)}")
    print(f"[Live-Task2] 输出目录: {Path(OUTPUT_DIR).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
