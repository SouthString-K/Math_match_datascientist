import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .llm import BaseTask1Model
from .pipeline import Task1Pipeline
from .schemas import DetectionResult, PreparedDocument, Task1Config


class Step1DetectionRunner:
    def __init__(self, model: BaseTask1Model, config: Task1Config):
        self.config = config
        self.pipeline = Task1Pipeline(model=model, config=config)

    def run(self, input_path: str, output_path: str, seed_path: Optional[str] = None) -> Dict[str, Any]:
        print(f"[Step1] 正在读取输入文件: {input_path}", flush=True)
        documents = self.pipeline.load_documents(input_path)
        print(f"[Step1] 已加载 {len(documents)} 条样本。", flush=True)
        seed_results, unmatched_seeds = self._load_seed_results(seed_path, documents) if seed_path else ({}, [])
        resumed_results, resumed_logs = self._load_resume_state(output_path, documents)
        if seed_results:
            print(f"[Step1] 已加载 {len(seed_results)} 条人工种子样本。", flush=True)
        if resumed_results:
            print(f"[Step1] 已从历史输出恢复 {len(resumed_results)} 条已分析样本。", flush=True)
        if unmatched_seeds:
            print(f"[Step1] 有 {len(unmatched_seeds)} 条种子样本未在当前全集中匹配到。", flush=True)
        initial_results: Dict[str, DetectionResult] = dict(resumed_results)
        initial_results.update(seed_results)
        print(
            f"[Step1] 半监督初始化完成：总样本 {len(documents)} 条，种子 {len(seed_results)} 条，已恢复 {len(resumed_results)} 条，待识别 {len(documents) - len(initial_results)} 条。",
            flush=True,
        )
        seed_samples = self._build_seed_samples(documents, seed_results)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        initial_payload = self._build_payload(
            documents=documents,
            detection_results=initial_results,
            logs=resumed_logs,
            pool_state={
                "status": "running",
                "round_pool_snapshots": [],
                "current_round_id": 0,
                "current_round_results": [],
                "parked_high_confidence_pool": [],
                "parked_low_confidence_pool": [],
                "final_high_confidence_pool": [],
                "final_candidate_pool": [],
                "final_low_confidence_pool": [],
            },
            seed_path=seed_path,
            seed_results=seed_results,
            unmatched_seeds=unmatched_seeds,
            total_documents=len(documents),
            seed_samples=seed_samples,
        )
        self._write_payload(output, initial_payload)
        detection_results, logs, pool_state = self.pipeline.run_detection(
            documents,
            seed_results=initial_results,
            progress_callback=lambda current_results, current_logs, current_pool_state: self._write_payload(
                output,
                self._build_payload(
                    documents=documents,
                    detection_results=current_results,
                    logs=resumed_logs + current_logs,
                    pool_state=current_pool_state,
                    seed_path=seed_path,
                    seed_results=seed_results,
                    unmatched_seeds=unmatched_seeds,
                    total_documents=len(documents),
                    seed_samples=seed_samples,
                ),
            ),
        )
        payload = self._build_payload(
            documents=documents,
            detection_results=detection_results,
            logs=resumed_logs + logs,
            pool_state=pool_state,
            seed_path=seed_path,
            seed_results=seed_results,
            unmatched_seeds=unmatched_seeds,
            total_documents=len(documents),
            seed_samples=seed_samples,
        )
        self._write_payload(output, payload)
        print(f"[Step1] JSON 标注结果已写出: {output}", flush=True)
        return payload

    def _build_payload(
        self,
        documents: List[PreparedDocument],
        detection_results: Dict[str, DetectionResult],
        logs: List,
        pool_state: Dict[str, Any],
        seed_path: Optional[str],
        seed_results: Dict[str, DetectionResult],
        unmatched_seeds: List[dict],
        total_documents: int,
        seed_samples: List[dict],
    ) -> Dict[str, Any]:
        grouped = self._group_annotations(documents, detection_results)
        bucketed = self._bucket_annotations(grouped)
        analyzed_count = len(detection_results)
        return {
            "config": {
                "seed_path": seed_path or "",
                "detection_confidence_threshold": self.config.detection_confidence_threshold,
                "detection_high_positive_threshold": self.config.detection_high_positive_threshold,
                "detection_high_negative_threshold": self.config.detection_high_negative_threshold,
                "detection_candidate_positive_threshold": self.config.detection_candidate_positive_threshold,
                "detection_candidate_negative_threshold": self.config.detection_candidate_negative_threshold,
                "max_detection_rounds": self.config.max_detection_rounds,
                "max_detection_pseudo_per_round": self.config.max_detection_pseudo_per_round,
            },
            "run_status": pool_state.get("status", "running"),
            "progress": {
                "total_documents": total_documents,
                "seed_count": len(seed_results),
                "analyzed_count": analyzed_count,
                "remaining_count": max(total_documents - analyzed_count, 0),
            },
            "seed_count": len(seed_results),
            "seed_samples": seed_samples,
            "unmatched_seed_count": len(unmatched_seeds),
            "initial_unlabeled_count": total_documents - len(seed_results),
            "pseudo_label_log": [log.to_dict() for log in logs],
            "pool_state": pool_state,
            "bucketed_results": bucketed,
            "grouped_results": grouped,
        }

    @staticmethod
    def _write_payload(output: Path, payload: Dict[str, Any]) -> None:
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _load_resume_state(
        output_path: str,
        documents: List[PreparedDocument],
    ) -> Tuple[Dict[str, DetectionResult], List[Any]]:
        output = Path(output_path)
        if not output.exists():
            return {}, []
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
        except Exception:
            return {}, []

        documents_by_id = {doc.document_id: doc for doc in documents}
        resumed_results: Dict[str, DetectionResult] = {}
        grouped = payload.get("grouped_results", {})
        if isinstance(grouped, dict):
            for items in grouped.values():
                if not isinstance(items, list):
                    continue
                for item in items:
                    document_id = str(item.get("sample_id", "")).strip()
                    if not document_id or document_id not in documents_by_id:
                        continue
                    resumed_results[document_id] = DetectionResult(
                        document_id=document_id,
                        is_event=int(item.get("is_event", 0)) == 1,
                        has_new_fact=int(item.get("has_new_fact", 0)),
                        has_market_impact_path=int(item.get("has_market_impact_path", 0)),
                        confidence=float(item.get("final_confidence", 0.0)),
                        seed_category=str(item.get("seed_category", "")),
                        reason=str(item.get("reason", "")),
                        label_source=str(item.get("label_source", "llm")),
                    )
        logs = payload.get("pseudo_label_log", [])
        if not isinstance(logs, list):
            logs = []
        return resumed_results, logs

    @staticmethod
    def _load_seed_results(
        seed_path: str,
        documents: List[PreparedDocument],
    ) -> Tuple[Dict[str, DetectionResult], List[dict]]:
        payload = json.loads(Path(seed_path).read_text(encoding="utf-8"))
        seed_results: Dict[str, DetectionResult] = {}
        unmatched_seeds: List[dict] = []
        documents_by_url = {
            Step1DetectionRunner._normalize_url(doc.url): doc
            for doc in documents
            if Step1DetectionRunner._normalize_url(doc.url)
        }
        documents_by_title = {
            (doc.title or "").strip(): doc
            for doc in documents
            if (doc.title or "").strip()
        }
        sample_groups = (
            payload.get("core_seed_samples", []),
            payload.get("auxiliary_seed_samples", []),
            payload.get("boundary_seed_samples", []),
            payload.get("high_confidence_samples", []),
            payload.get("medium_confidence_samples", []),
            payload.get("low_confidence_samples", []),
        )
        for group in sample_groups:
            for item in group:
                matched_doc = None
                raw_url = Step1DetectionRunner._normalize_url(str(item.get("url", "")))
                if raw_url:
                    matched_doc = documents_by_url.get(raw_url)
                if matched_doc is None:
                    raw_title = str(item.get("title", "")).strip()
                    matched_doc = documents_by_title.get(raw_title)
                if matched_doc is None:
                    unmatched_seeds.append(item)
                    continue
                seed_results[matched_doc.document_id] = DetectionResult(
                    document_id=matched_doc.document_id,
                    is_event=int(item.get("is_event", 0)) == 1,
                    has_new_fact=int(item.get("has_new_fact", 0)),
                    has_market_impact_path=int(item.get("has_market_impact_path", 0)),
                    confidence=float(item.get("final_confidence", 1.0)),
                    seed_category=str(item.get("seed_category", "high_confidence_positive" if int(item.get("is_event", 0)) == 1 else "high_confidence_negative")),
                    reason=str(item.get("reason", "人工种子标注")),
                    label_source="human_seed",
                )
        return seed_results, unmatched_seeds

    @staticmethod
    def _normalize_url(url: str) -> str:
        return (url or "").strip()

    @staticmethod
    def _build_seed_samples(
        documents: List[PreparedDocument],
        seed_results: Dict[str, DetectionResult],
    ) -> List[dict]:
        documents_by_id = {doc.document_id: doc for doc in documents}
        items: List[dict] = []
        for document_id, result in seed_results.items():
            doc = documents_by_id[document_id]
            items.append(
                {
                    "sample_id": doc.document_id,
                    "source_site": doc.source_site,
                    "title": doc.title,
                    "url": doc.url,
                    "publish_time": doc.publish_time,
                    "is_event": 1 if result.is_event else 0,
                    "has_new_fact": int(result.has_new_fact),
                    "has_market_impact_path": int(result.has_market_impact_path),
                    "reason": result.reason,
                    "final_confidence": round(float(result.confidence), 2),
                    "seed_category": result.seed_category,
                    "label_source": result.label_source,
                }
            )
        return sorted(items, key=lambda item: item["sample_id"])

    @staticmethod
    def _group_annotations(
        documents: List[PreparedDocument],
        detection_results: Dict[str, DetectionResult],
    ) -> Dict[str, List[dict]]:
        grouped: Dict[str, List[dict]] = OrderedDict()
        for doc in documents:
            if doc.document_id not in detection_results:
                continue
            result = detection_results[doc.document_id]
            site = doc.source_site or "unknown"
            grouped.setdefault(site, []).append(
                {
                    "sample_id": doc.document_id,
                    "source_site": site,
                    "title": doc.title,
                    "url": doc.url,
                    "is_event": 1 if result.is_event else 0,
                    "has_new_fact": int(result.has_new_fact),
                    "has_market_impact_path": int(result.has_market_impact_path),
                    "reason": result.reason,
                    "final_confidence": round(float(result.confidence), 2),
                    "label_source": result.label_source,
                    "sample_bucket": Step1DetectionRunner._resolve_bucket(result),
                    "seed_category": result.seed_category,
                }
            )
        return grouped

    @staticmethod
    def _resolve_bucket(result: DetectionResult) -> str:
        return result.seed_category

    @staticmethod
    def _bucket_annotations(grouped: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
        bucketed: Dict[str, List[dict]] = OrderedDict(
            (
                ("high_confidence_positive", []),
                ("high_confidence_negative", []),
                ("medium_confidence_positive", []),
                ("medium_confidence_negative", []),
                ("boundary_positive", []),
                ("boundary_negative", []),
            )
        )
        for items in grouped.values():
            for item in items:
                bucketed.setdefault(item["sample_bucket"], []).append(item)
        return bucketed
