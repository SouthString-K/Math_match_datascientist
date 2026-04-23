"""
实测分类脚本：东方财富30条新闻 → 跳过检测，直接分类
- 读取 Web_scrope/东方财富30.json（只有 title + url）
- 补全 is_event=1 / publish_time / source_site 等字段
- 调用 ClassificationOnlyRunner 直接走分类流程
"""
import json
import os
import sys
from pathlib import Path

from task1.classification_only import ClassificationOnlyRunner
from task1.schemas import Task1Config

# ---------- 配置 ----------
RAW_INPUT = r"D:\Math_match\codes\Web_scrope\东方财富30.json"
OUTPUT_DIR = r"D:\Math_match\codes\outputs\live_classification_eastmoney30"

MODEL = "qwen3-max"
API_KEY = "sk-a0841f92a8eb48bdab73457cf8227229"
ENABLE_THINKING = False

# 发布日期（实测当天）
PUBLISH_TIME = "4.20"
SOURCE_SITE = "东方财富"


def prepare_input(raw_path: str, output_dir: str) -> str:
    """
    将原始 JSON（title+url）补全为分类所需的格式
    返回补全后的 JSON 文件路径
    """
    raw_items = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    enriched = []
    for idx, item in enumerate(raw_items, 1):
        enriched.append({
            "sample_id": f"LIVE-{idx:04d}",
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": "",               # 分类 Prompt 里 LLM 会自己访问 URL
            "publish_time": PUBLISH_TIME,
            "source_site": SOURCE_SITE,
            "is_event": 1,               # 跳过检测，全部视为事件
            "has_new_fact": 1,
            "has_market_impact_path": 1,
            "final_confidence": 1.0,
            "seed_category": "high_confidence_positive",
            "label_source": "upstream_detection",
            "reason": "实测输入，跳过检测阶段",
        })

    out_path = Path(output_dir) / "input_enriched.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[Live] 已补全 {len(enriched)} 条事件 → {out_path}", flush=True)
    return str(out_path)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    os.environ["DASHSCOPE_API_KEY"] = API_KEY

    # Step 1: 补全输入数据
    input_path = prepare_input(RAW_INPUT, OUTPUT_DIR)

    # Step 2: 配置分类运行器
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

    # Step 3: 执行分类
    code = runner.run(input_path=input_path, output_dir=OUTPUT_DIR)
    print(f"[Live] 输出目录: {Path(OUTPUT_DIR).resolve()}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
