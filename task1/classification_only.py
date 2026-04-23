import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

from .llm import DashScopeTask1Model, MockTask1Model
from .pipeline import Task1Pipeline
from .schemas import DetectionResult, PreparedDocument, Task1Config


class ClassificationOnlyRunner:
    def __init__(self, config: Task1Config, provider: str, model_name: str, api_key: str = ""):
        self.config = config
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key

    def _build_model(self):
        if self.provider == "mock":
            return MockTask1Model()
        return DashScopeTask1Model(
            model=self.model_name,
            api_key=self.api_key or None,
            enable_thinking=self.config.enable_thinking,
        )

    @staticmethod
    def _build_detection_results(documents: List[PreparedDocument]) -> Dict[str, DetectionResult]:
        detection_results: Dict[str, DetectionResult] = {}
        for doc in documents:
            is_event = bool(doc.event_label) if doc.event_label is not None else True
            confidence = doc.detection_confidence if doc.detection_confidence is not None else 1.0
            seed_category = doc.detection_seed_category or (
                "high_confidence_positive" if is_event else "high_confidence_negative"
            )
            detection_results[doc.document_id] = DetectionResult(
                document_id=doc.document_id,
                is_event=is_event,
                has_new_fact=doc.has_new_fact if doc.has_new_fact is not None else (1 if is_event else 0),
                has_market_impact_path=doc.has_market_impact_path
                if doc.has_market_impact_path is not None
                else (1 if is_event else 0),
                confidence=float(confidence),
                seed_category=seed_category,
                reason=doc.detection_reason or "来自事件识别阶段输入",
                label_source=doc.detection_label_source or "upstream_detection",
            )
        return detection_results

    @staticmethod
    def _skip_file(output_dir: str) -> Path:
        return Path(output_dir) / "classification_skipped.json"

    def _load_skipped_records(self, output_dir: str) -> List[dict]:
        path = self._skip_file(output_dir)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return data if isinstance(data, list) else []

    def _write_skipped_records(self, output_dir: str, records: List[dict]) -> None:
        path = self._skip_file(output_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _extract_failed_document_id(message: str) -> str:
        patterns = [
            r"Classification failed for\s+([A-Za-z0-9_-]+):",
            r"for\s+(DOC-\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                return match.group(1)
        return ""

    def run(self, input_path: str, output_dir: str) -> int:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
        if self.api_key:
            os.environ["DASHSCOPE_API_KEY"] = self.api_key

        model = self._build_model()
        pipeline = Task1Pipeline(model=model, config=self.config)

        print(f"[Classification] 正在读取输入文件: {input_path}", flush=True)
        all_documents = pipeline.load_documents(input_path)
        print(f"[Classification] 已加载 {len(all_documents)} 条事件样本。", flush=True)
        documents_by_id = {doc.document_id: doc for doc in all_documents}

        while True:
            skipped_records = self._load_skipped_records(output_dir)
            skipped_ids = {str(item.get("sample_id") or "").strip() for item in skipped_records}
            active_documents = [doc for doc in all_documents if doc.document_id not in skipped_ids]
            if skipped_ids:
                print(
                    f"[Classification] 已跳过 {len(skipped_ids)} 条报错样本，当前继续处理 {len(active_documents)} 条。",
                    flush=True,
                )

            detection_results = self._build_detection_results(active_documents)
            existing_results, existing_logs = pipeline._load_existing_classification_state(output_dir)
            if existing_results:
                print(
                    f"[Classification] 已恢复 {len(existing_results)} 条已完成分类结果，将从断点继续。",
                    flush=True,
                )

            def progress_callback(current_results, current_logs):
                pipeline._export_classification_progress(
                    output_dir=output_dir,
                    documents=active_documents,
                    detection_results=detection_results,
                    classification_results=current_results,
                    detection_logs=[],
                    classification_logs=existing_logs + current_logs,
                )

            try:
                classification_results, classification_logs = pipeline.run_classification(
                    documents=active_documents,
                    detection_results=detection_results,
                    existing_results=existing_results,
                    progress_callback=progress_callback,
                )
            except Exception as exc:
                message = str(exc)
                failed_id = self._extract_failed_document_id(message)
                if not failed_id or failed_id in skipped_ids or failed_id not in documents_by_id:
                    raise
                failed_doc = documents_by_id[failed_id]
                skipped_records.append(
                    {
                        "sample_id": failed_doc.document_id,
                        "title": failed_doc.title,
                        "url": failed_doc.url,
                        "source_site": failed_doc.source_site,
                        "error": message,
                    }
                )
                self._write_skipped_records(output_dir, skipped_records)
                print(
                    f"[Classification] 样本 {failed_id} 分类失败，已加入跳过列表，继续处理后续样本。",
                    flush=True,
                )
                continue

            pipeline._export_classification_progress(
                output_dir=output_dir,
                documents=active_documents,
                detection_results=detection_results,
                classification_results=classification_results,
                detection_logs=[],
                classification_logs=existing_logs + classification_logs,
            )

            print(f"[Classification] 分类完成，共输出 {len(classification_results)} 条结果。", flush=True)
            print(f"[Classification] 输出目录: {Path(output_dir).resolve()}", flush=True)
            if skipped_records:
                print(
                    f"[Classification] 另有 {len(skipped_records)} 条失败样本已写入跳过文件。",
                    flush=True,
                )
            return 0
