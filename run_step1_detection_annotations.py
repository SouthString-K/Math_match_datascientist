import os
import sys
from pathlib import Path

from task1.llm import DashScopeTask1Model, MockTask1Model
from task1.schemas import Task1Config
from task1.step1_detection import Step1DetectionRunner

# Annotation-pool generation for new seed building
INPUT_PATH = r"D:\Math_match\codes\Web_scrope\东方财富后400.json"
OUTPUT_PATH = r"D:\Math_match\codes\outputs\step1_detection_annotations_eastmoney400.json"
PROVIDER = "dashscope"  # "mock" or "dashscope"
MODEL = "qwen3-max"
API_KEY = "sk-a0841f92a8eb48bdab73457cf8227229"
MAX_DETECTION_ROUNDS = 1
DETECTION_THRESHOLD = 0.90
HIGH_POSITIVE_THRESHOLD = 0.90
HIGH_NEGATIVE_THRESHOLD = 0.10
CANDIDATE_POSITIVE_THRESHOLD = 0.70
CANDIDATE_NEGATIVE_THRESHOLD = 0.30
MAX_PSEUDO_PER_ROUND = 30
ENABLE_THINKING = False


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    os.environ["DASHSCOPE_API_KEY"] = API_KEY

    config = Task1Config(
        max_detection_rounds=MAX_DETECTION_ROUNDS,
        detection_confidence_threshold=DETECTION_THRESHOLD,
        detection_high_positive_threshold=HIGH_POSITIVE_THRESHOLD,
        detection_high_negative_threshold=HIGH_NEGATIVE_THRESHOLD,
        detection_candidate_positive_threshold=CANDIDATE_POSITIVE_THRESHOLD,
        detection_candidate_negative_threshold=CANDIDATE_NEGATIVE_THRESHOLD,
        max_detection_pseudo_per_round=MAX_PSEUDO_PER_ROUND,
        enable_thinking=ENABLE_THINKING,
    )

    if PROVIDER == "mock":
        model = MockTask1Model()
    else:
        model = DashScopeTask1Model(
            model=MODEL,
            api_key=API_KEY,
            enable_thinking=config.enable_thinking,
        )

    runner = Step1DetectionRunner(model=model, config=config)
    payload = runner.run(input_path=INPUT_PATH, output_path=OUTPUT_PATH, seed_path=None)
    grouped = payload["grouped_results"]
    count = sum(len(items) for items in grouped.values())
    print(f"Prepared annotation pool for {count} samples.")
    print(f"Outputs written to {Path(OUTPUT_PATH).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
