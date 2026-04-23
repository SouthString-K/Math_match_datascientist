import argparse
import sys

from .llm import DashScopeTask2Model, MockTask2Model
from .pipeline import Task2Pipeline
from .schemas import Task2Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 2 company profiling and event-company matching")
    parser.add_argument("--company-input", required=True, help="HS300 company list JSON path")
    parser.add_argument("--events-input", required=True, help="Task 1 classification results JSON path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--stage", choices=["profile", "match", "all"], default="all")
    parser.add_argument("--provider", choices=["mock", "dashscope"], default="mock")
    parser.add_argument("--model", default="qwen3-max")
    parser.add_argument("--profile-batch-size", type=int, default=1)
    parser.add_argument("--match-candidate-limit", type=int, default=12)
    parser.add_argument("--max-match-results", type=int, default=5)
    parser.add_argument("--min-recall-score", type=float, default=1.0)
    parser.add_argument("--min-direct-score", type=float, default=3.0)
    parser.add_argument("--max-companies", type=int, default=0)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--disable-thinking", action="store_true")
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    config = Task2Config(
        profile_batch_size=args.profile_batch_size,
        match_candidate_limit=args.match_candidate_limit,
        max_match_results=args.max_match_results,
        min_recall_score=args.min_recall_score,
        min_direct_score=args.min_direct_score,
        max_companies=args.max_companies,
        max_events=args.max_events,
        enable_thinking=not args.disable_thinking,
    )

    if args.provider == "mock":
        model = MockTask2Model()
    else:
        model = DashScopeTask2Model(model=args.model, enable_thinking=config.enable_thinking)

    pipeline = Task2Pipeline(model=model, config=config)
    result = pipeline.run(
        company_input=args.company_input,
        events_input=args.events_input,
        output_dir=args.output,
        stage=args.stage,
    )

    if "company_profiles" in result:
        print(f"[Task2] 已生成 {len(result['company_profiles'])} 家公司画像。")
    if "links" in result:
        print(f"[Task2] 已完成 {len(result['links'])} 条事件的公司匹配。")
    print(f"[Task2] 输出目录: {args.output}")
    return 0
