import os
import sys
from pathlib import Path

from task1.classification_only import ClassificationOnlyRunner
from task1.schemas import Task1Config


INPUT_PATH = r"D:\Math_match\codes\outputs\step1_event_only_for_classification_eastmoney400.json"
OUTPUT_DIR = r"D:\Math_match\codes\outputs\classification_run_eastmoney400_v2"
MODEL = "qwen3-max"
API_KEY = "sk-a0841f92a8eb48bdab73457cf8227229"
ENABLE_THINKING = False


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    os.environ["DASHSCOPE_API_KEY"] = API_KEY

    config = Task1Config(
        max_classification_rounds=1,
        enable_thinking=ENABLE_THINKING,
    )
    runner = ClassificationOnlyRunner(
        config=config,
        provider="dashscope",
        model_name=MODEL,
        api_key=API_KEY,
    )
    code = runner.run(input_path=INPUT_PATH, output_dir=OUTPUT_DIR)
    print(f"Outputs written to {Path(OUTPUT_DIR).resolve()}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
