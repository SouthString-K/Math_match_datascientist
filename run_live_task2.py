"""
实测 Task2：公司关联匹配（match 阶段）
- 复用已有的 300 家公司画像
- 输入 Task1 的分类结果
- 输出 event_company_links + assoc_matrix
"""
import os
import sys
from pathlib import Path

from task2.llm import DashScopeTask2Model
from task2.pipeline import Task2Pipeline
from task2.schemas import Task2Config

# ---------- 配置 ----------
COMPANY_PROFILES = r"D:\Math_match\codes\outputs\task2_profile_run\company_profiles.json"
EVENTS_INPUT = r"D:\Math_match\codes\outputs\live_classification_eastmoney30\classification_results.json"
OUTPUT_DIR = r"D:\Math_match\codes\outputs\live_task2_match"

MODEL = "qwen3-max"
API_KEY = "sk-a0841f92a8eb48bdab73457cf8227229"
ENABLE_THINKING = False

# 匹配参数
MATCH_CANDIDATE_LIMIT = 12
MAX_MATCH_RESULTS = 5
MIN_RECALL_SCORE = 1.0
MIN_DIRECT_SCORE = 3.0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    os.environ["DASHSCOPE_API_KEY"] = API_KEY

    config = Task2Config(
        profile_batch_size=1,
        match_candidate_limit=MATCH_CANDIDATE_LIMIT,
        max_match_results=MAX_MATCH_RESULTS,
        min_recall_score=MIN_RECALL_SCORE,
        min_direct_score=MIN_DIRECT_SCORE,
        max_companies=0,
        max_events=0,
        enable_thinking=ENABLE_THINKING,
    )

    model = DashScopeTask2Model(
        model=MODEL,
        api_key=API_KEY,
        enable_thinking=config.enable_thinking,
    )

    # 手动加载画像和事件，然后直接跑 match
    pipeline = Task2Pipeline(model=model, config=config)

    # 加载已有画像
    profiles = pipeline.load_profiles(COMPANY_PROFILES)
    print(f"[Live-Task2] 已加载 {len(profiles)} 家公司画像", flush=True)

    # 加载事件
    events = pipeline.load_events(EVENTS_INPUT)
    print(f"[Live-Task2] 已加载 {len(events)} 条事件", flush=True)

    # 将画像复制到输出目录，以便 pipeline 内部查找
    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    import json
    import shutil
    # 复制画像到输出目录
    shutil.copy2(COMPANY_PROFILES, output / "company_profiles.json")
    # 复制 seeds
    hs300_src = Path(r"D:\Math_match\codes\outputs\task2_profile_run\normalized_hs300.json")
    if hs300_src.exists():
        shutil.copy2(str(hs300_src), output / "normalized_hs300.json")

    # 运行 match
    links = pipeline.match_events_to_companies(events, profiles, OUTPUT_DIR)

    # 输出统计
    matched = sum(1 for link in links if link.matched_companies)
    total_matches = sum(len(link.matched_companies) for link in links)
    print(f"[Live-Task2] 共完成 {len(links)} 条事件→公司匹配，其中 {matched} 条有匹配结果，共 {total_matches} 个匹配对", flush=True)
    print(f"[Live-Task2] 输出目录: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
