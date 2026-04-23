import argparse
from pathlib import Path

from .llm import DashScopeTask1Model, MockTask1Model
from .pipeline import Task1Pipeline
from .schemas import Task1Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 1 event pipeline")
    parser.add_argument("--input", required=True, help="Input CSV/JSONL/XLSX path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--provider", choices=["mock", "dashscope"], default="mock")
    parser.add_argument("--model", default="qwen3-max")
    parser.add_argument("--max-detection-rounds", type=int, default=3)
    parser.add_argument("--max-classification-rounds", type=int, default=3)
    parser.add_argument("--detection-threshold", type=float, default=0.82)
    parser.add_argument("--classification-threshold", type=float, default=0.80)
    parser.add_argument("--max-pseudo-per-round", type=int, default=25)
    parser.add_argument("--max-pseudo-per-class", type=int, default=4)
    parser.add_argument("--disable-thinking", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = Task1Config(
        max_detection_rounds=args.max_detection_rounds,
        max_classification_rounds=args.max_classification_rounds,
        detection_confidence_threshold=args.detection_threshold,
        classification_confidence_threshold=args.classification_threshold,
        max_detection_pseudo_per_round=args.max_pseudo_per_round,
        max_classification_pseudo_per_class=args.max_pseudo_per_class,
        enable_thinking=not args.disable_thinking,
    )

    if args.provider == "mock":
        model = MockTask1Model()
    else:
        model = DashScopeTask1Model(model=args.model, enable_thinking=config.enable_thinking)

    pipeline = Task1Pipeline(model=model, config=config)
    result = pipeline.run(input_path=args.input, output_dir=args.output)

    event_docs = sum(1 for item in result["detection_results"].values() if item.is_event)
    print(f"Processed {len(result['documents'])} documents.")
    print(f"Detected {event_docs} event documents.")
    print(f"Extracted {len(result['events'])} structured events.")
    print(f"Outputs written to {Path(args.output).resolve()}")
    return 0
