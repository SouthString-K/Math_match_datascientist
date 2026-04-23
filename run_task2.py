import os
import sys
from pathlib import Path

from task2.llm import DashScopeTask2Model, MockTask2Model
from task2.pipeline import Task2Pipeline
from task2.schemas import Task2Config


# Task 2 step-1 profile run configuration
# This file is configured for: company profile generation only.
COMPANY_INPUT = r"D:\Math_match\codes\task2\hushen300.json"
EVENTS_INPUT = r"D:\Math_match\codes\outputs\classification_run_eastmoney400_v2\classification_results.json"
OUTPUT_DIR = r"D:\Math_match\codes\outputs\task2_profile_run"

# stage options: profile / match / all
STAGE = "profile"
PROVIDER = "dashscope"  # mock / dashscope
MODEL = "qwen3-max"
API_KEY = "sk-a0841f92a8eb48bdab73457cf8227229"  # ???????? DASHSCOPE_API_KEY????????????????????

# Runtime controls
PROFILE_BATCH_SIZE = 1
MATCH_CANDIDATE_LIMIT = 12
MAX_MATCH_RESULTS = 5
MIN_RECALL_SCORE = 1.0
MIN_DIRECT_SCORE = 3.0
MAX_COMPANIES = 0  # 0 ???? 300 ?
MAX_EVENTS = 0
ENABLE_THINKING = False


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    api_key = API_KEY.strip() or os.getenv("DASHSCOPE_API_KEY", "").strip()
    if PROVIDER == "dashscope" and not api_key:
        raise ValueError("??? run_task2.py ??? API_KEY???????? DASHSCOPE_API_KEY?")

    config = Task2Config(
        profile_batch_size=PROFILE_BATCH_SIZE,
        match_candidate_limit=MATCH_CANDIDATE_LIMIT,
        max_match_results=MAX_MATCH_RESULTS,
        min_recall_score=MIN_RECALL_SCORE,
        min_direct_score=MIN_DIRECT_SCORE,
        max_companies=MAX_COMPANIES,
        max_events=MAX_EVENTS,
        enable_thinking=ENABLE_THINKING,
    )

    if PROVIDER == "mock":
        model = MockTask2Model()
    else:
        model = DashScopeTask2Model(
            model=MODEL,
            api_key=api_key,
            enable_thinking=config.enable_thinking,
        )

    pipeline = Task2Pipeline(model=model, config=config)
    result = pipeline.run(
        company_input=COMPANY_INPUT,
        events_input=EVENTS_INPUT,
        output_dir=OUTPUT_DIR,
        stage=STAGE,
    )

    if "company_profiles" in result:
        print(f"[Task2] ??? {len(result['company_profiles'])} ??????")
    if "links" in result:
        print(f"[Task2] ??? {len(result['links'])} ?????????")
    print(f"[Task2] ????: {Path(OUTPUT_DIR).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
