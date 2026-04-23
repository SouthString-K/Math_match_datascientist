import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .llm import BaseTask1Model
from .preprocessing import build_combined_text, normalize_text
from .schemas import (
    ClassificationResult,
    DetectionResult,
    PreparedDocument,
    PseudoLabelLog,
    StructuredEvent,
    Task1Config,
)


class Task1Pipeline:
    def __init__(self, model: BaseTask1Model, config: Task1Config):
        self.model = model
        self.config = config

    def load_documents(self, input_path: str) -> List[PreparedDocument]:
        path = Path(input_path)
        if path.is_dir():
            return self._load_documents_from_directory(path)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".jsonl":
            df = pd.read_json(path, lines=True)
        elif path.suffix.lower() == ".json":
            return self._load_documents_from_json(path)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported input format: {path.suffix}")

        documents: List[PreparedDocument] = []
        for idx, row in df.fillna("").iterrows():
            document_id = str(row.get("document_id") or row.get("id") or f"DOC-{idx + 1:04d}")
            title = str(row.get("title") or row.get("headline") or "")
            content = str(row.get("content") or row.get("body") or row.get("text") or row.get("html") or "")
            cleaned = normalize_text(content)
            documents.append(
                PreparedDocument(
                    document_id=str(row.get("sample_id") or document_id),
                    title=title.strip(),
                    content=content,
                    cleaned_content=cleaned,
                    combined_text=build_combined_text(title, cleaned, self.config.max_content_chars),
                    url=str(row.get("url") or row.get("link") or ""),
                    publish_time=str(row.get("publish_time") or row.get("date") or row.get("time") or ""),
                    source_site=str(row.get("source_site") or row.get("source") or ""),
                    event_label=self._parse_optional_int(row.get("event_label") if row.get("event_label") not in ("", None) else row.get("is_event")),
                    event_type=str(row.get("event_type") or "").strip(),
                    has_new_fact=self._parse_optional_int(row.get("has_new_fact")),
                    has_market_impact_path=self._parse_optional_int(row.get("has_market_impact_path")),
                    detection_reason=str(row.get("reason") or ""),
                    detection_confidence=self._parse_optional_float(row.get("final_confidence")),
                    detection_seed_category=str(row.get("seed_category") or row.get("sample_bucket") or ""),
                    detection_label_source=str(row.get("label_source") or ""),
                )
            )
        return documents

    def _load_documents_from_directory(self, path: Path) -> List[PreparedDocument]:
        documents: List[PreparedDocument] = []
        counter = 1
        json_files = sorted(
            file_path
            for file_path in path.glob("*.json")
            if file_path.name != "sampled_10_each.json"
        )
        if not json_files:
            raise ValueError(f"No JSON files found in directory: {path}")

        for file_path in json_files:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            file_documents, counter = self._parse_json_payload(
                payload=payload,
                default_source_site=file_path.stem,
                counter_start=counter,
            )
            documents.extend(file_documents)
        return documents

    def _load_documents_from_json(self, path: Path) -> List[PreparedDocument]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("grouped_results"), dict):
            payload = payload["grouped_results"]
        documents, _ = self._parse_json_payload(payload=payload, default_source_site=path.stem, counter_start=1)
        return documents

    def _parse_json_payload(
        self,
        payload,
        default_source_site: str,
        counter_start: int,
    ) -> Tuple[List[PreparedDocument], int]:
        documents: List[PreparedDocument] = []

        if isinstance(payload, list):
            groups = [(default_source_site or "json", payload)]
        elif isinstance(payload, dict):
            groups = list(payload.items())
        else:
            raise ValueError("Unsupported JSON structure; expected list or dict")

        counter = counter_start
        for group_name, items in groups:
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "")
                content = str(item.get("content") or item.get("body") or item.get("text") or item.get("html") or "")
                cleaned = normalize_text(content)
                documents.append(
                    PreparedDocument(
                        document_id=str(item.get("document_id") or item.get("sample_id") or item.get("id") or f"DOC-{counter:04d}"),
                        title=title.strip(),
                        content=content,
                        cleaned_content=cleaned,
                        combined_text=build_combined_text(title, cleaned, self.config.max_content_chars),
                        url=str(item.get("url") or item.get("link") or item.get("original_href") or ""),
                        publish_time=str(item.get("publish_time") or item.get("date") or item.get("time") or ""),
                        source_site=str(item.get("source_site") or item.get("source") or group_name or default_source_site or ""),
                        event_label=self._parse_optional_int(item.get("event_label") if item.get("event_label") not in ("", None) else item.get("is_event")),
                        event_type=str(item.get("event_type") or "").strip(),
                        has_new_fact=self._parse_optional_int(item.get("has_new_fact")),
                        has_market_impact_path=self._parse_optional_int(item.get("has_market_impact_path")),
                        detection_reason=str(item.get("reason") or ""),
                        detection_confidence=self._parse_optional_float(item.get("final_confidence")),
                        detection_seed_category=str(item.get("seed_category") or item.get("sample_bucket") or ""),
                        detection_label_source=str(item.get("label_source") or ""),
                    )
                )
                counter += 1
        return documents, counter

    @staticmethod
    def _parse_optional_int(value):
        if value in ("", None):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_optional_float(value):
        if value in ("", None):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _fewshot_detection_examples(
        self,
        documents_by_id: Dict[str, PreparedDocument],
        results: Dict[str, DetectionResult],
    ) -> List[dict]:
        priority_order = {
            "high_confidence_positive": 0,
            "high_confidence_negative": 1,
            "medium_confidence_positive": 2,
            "medium_confidence_negative": 3,
            "boundary_positive": 4,
            "boundary_negative": 5,
        }
        positives = []
        negatives = []
        for document_id, result in results.items():
            doc = documents_by_id[document_id]
            example = {
                "sample_id": document_id,
                "title": doc.title,
                "url": doc.url,
                "label": result.seed_category or ("候选金融事件" if result.is_event else "非事件"),
                "reason": result.reason,
                "final_confidence": round(float(result.confidence), 2),
            }
            if result.is_event:
                positives.append((priority_order.get(result.seed_category, 99), -float(result.confidence), example))
            else:
                negatives.append((priority_order.get(result.seed_category, 99), float(result.confidence), example))
        positives = [item[2] for item in sorted(positives, key=lambda item: (item[0], item[1], item[2]["sample_id"]))]
        negatives = [item[2] for item in sorted(negatives, key=lambda item: (item[0], item[1], item[2]["sample_id"]))]
        half = max(1, self.config.max_fewshot_examples // 2)
        return positives[:half] + negatives[:half]

    def _fewshot_classification_examples(
        self,
        documents_by_id: Dict[str, PreparedDocument],
        results: Dict[str, ClassificationResult],
    ) -> List[dict]:
        grouped: Dict[str, List[dict]] = {}
        for document_id, result in results.items():
            doc = documents_by_id[document_id]
            grouped.setdefault(result.event_type, []).append(
                {
                    "title": doc.title,
                    "text": doc.cleaned_content[:200],
                    "label": result.event_type,
                }
            )
        examples: List[dict] = []
        per_class = max(1, self.config.max_fewshot_examples // max(len(grouped), 1))
        for label in sorted(grouped):
            examples.extend(grouped[label][:per_class])
        return examples[: self.config.max_fewshot_examples]

    @staticmethod
    def _result_with_document(doc: PreparedDocument, result: DetectionResult) -> Dict[str, Any]:
        return {
            "sample_id": doc.document_id,
            "source_site": doc.source_site,
            "title": doc.title,
            "url": doc.url,
            "publish_time": doc.publish_time,
            "is_event": 1 if result.is_event else 0,
            "has_new_fact": int(result.has_new_fact),
            "has_market_impact_path": int(result.has_market_impact_path),
            "final_confidence": round(float(result.confidence), 2),
            "seed_category": result.seed_category,
            "reason": result.reason,
            "label_source": result.label_source,
        }

    def _render_detection_pool(
        self,
        documents_by_id: Dict[str, PreparedDocument],
        results: List[DetectionResult],
    ) -> List[Dict[str, Any]]:
        return [
            self._result_with_document(documents_by_id[result.document_id], result)
            for result in results
        ]

    @staticmethod
    def _classification_result_with_document(
        doc: PreparedDocument,
        detection_result: DetectionResult,
        classification_result: ClassificationResult,
    ) -> Dict[str, Any]:
        return {
            "sample_id": doc.document_id,
            "source_site": doc.source_site,
            "title": doc.title,
            "url": doc.url,
            "publish_time": doc.publish_time,
            "is_event": 1 if detection_result.is_event else 0,
            "has_new_fact": int(detection_result.has_new_fact),
            "has_market_impact_path": int(detection_result.has_market_impact_path),
            "detection_confidence": round(float(detection_result.confidence), 2),
            "detection_seed_category": detection_result.seed_category,
            "detection_reason": detection_result.reason,
            "detection_label_source": detection_result.label_source,
            "event_type": classification_result.event_type,
            "duration_type": classification_result.duration_type,
            "event_summary": classification_result.event_summary,
            "heat": round(float(classification_result.heat), 3),
            "heat_signal": classification_result.heat_signal,
            "heat_comment_count": classification_result.heat_comment_count,
            "keyword": classification_result.keyword,
            "keyword_score": int(classification_result.keyword_score),
            "scale_score": round(float(classification_result.scale_score), 3),
            "event_intensity": round(float(classification_result.event_intensity), 3),
            "influence_range": round(float(classification_result.influence_range), 3),
            "attribute_score": round(float(classification_result.attribute_score), 3),
            "classification_confidence": round(float(classification_result.confidence), 2),
            "classification_reason": classification_result.reason,
            "classification_label_source": classification_result.label_source,
        }

    def run_detection(
        self,
        documents: List[PreparedDocument],
        seed_results: Optional[Dict[str, DetectionResult]] = None,
        progress_callback: Optional[Callable[[Dict[str, DetectionResult], List[PseudoLabelLog], Dict[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, DetectionResult], List[PseudoLabelLog], Dict[str, Any]]:
        documents_by_id = {doc.document_id: doc for doc in documents}
        labeled: Dict[str, DetectionResult] = {}
        parked_low_results: Dict[str, DetectionResult] = {}
        parked_high_results: Dict[str, DetectionResult] = {}
        unlabeled: List[PreparedDocument] = []
        logs: List[PseudoLabelLog] = []
        round_pool_snapshots: List[Dict[str, Any]] = []
        seed_results = seed_results or {}

        for doc in documents:
            if doc.document_id in seed_results:
                labeled[doc.document_id] = seed_results[doc.document_id]
            elif doc.event_label is None:
                unlabeled.append(doc)
            else:
                labeled[doc.document_id] = DetectionResult(
                    document_id=doc.document_id,
                    is_event=bool(doc.event_label),
                    has_new_fact=1 if bool(doc.event_label) else 0,
                    has_market_impact_path=1 if bool(doc.event_label) else 0,
                    confidence=1.0,
                    seed_category="high_confidence_positive" if bool(doc.event_label) else "high_confidence_negative",
                    reason="人工种子标注",
                    label_source="human_seed",
                )
        self._emit_detection_progress(
            progress_callback=progress_callback,
            documents_by_id=documents_by_id,
            labeled=labeled,
            parked_high_results=parked_high_results,
            parked_low_results=parked_low_results,
            current_round_results=[],
            logs=logs,
            round_pool_snapshots=round_pool_snapshots,
            status="running",
            current_round_id=0,
        )

        if not labeled:
            total = len(documents)
            print(f"[Step1] 开始识别，共 {total} 条样本。", flush=True)
            for idx, doc in enumerate(documents, 1):
                print(
                    f"[Step1] 正在处理 {idx}/{total}: {doc.document_id} | {doc.source_site} | {doc.title[:50]}",
                    flush=True,
                )
                labeled[doc.document_id] = self.model.detect(doc, examples=[])
                self._emit_detection_progress(
                    progress_callback=progress_callback,
                    documents_by_id=documents_by_id,
                    labeled=labeled,
                    parked_high_results=parked_high_results,
                    parked_low_results=parked_low_results,
                    current_round_results=[],
                    logs=logs,
                    round_pool_snapshots=round_pool_snapshots,
                    status="running",
                    current_round_id=0,
                )
            print("[Step1] 识别完成。", flush=True)
            return labeled, logs, {"round_pool_snapshots": round_pool_snapshots, "final_candidate_pool": [], "final_low_confidence_pool": []}

        for round_id in range(1, self.config.max_detection_rounds + 1):
            if not unlabeled:
                break
            print(
                f"[Step1] 第 {round_id} 轮识别开始，待处理 {len(unlabeled)} 条未标注样本。",
                flush=True,
            )
            examples = self._fewshot_detection_examples(documents_by_id, labeled)
            candidates = []
            total = len(unlabeled)
            for idx, doc in enumerate(unlabeled, 1):
                print(
                    f"[Step1] 第 {round_id} 轮处理中 {idx}/{total}: {doc.document_id} | {doc.source_site} | {doc.title[:50]}",
                    flush=True,
                )
                candidates.append(self.model.detect(doc, examples=examples))
                self._emit_detection_progress(
                    progress_callback=progress_callback,
                    documents_by_id=documents_by_id,
                    labeled=labeled,
                    parked_high_results=parked_high_results,
                    parked_low_results=parked_low_results,
                    current_round_results=candidates,
                    logs=logs,
                    round_pool_snapshots=round_pool_snapshots,
                    status="running",
                    current_round_id=round_id,
                )

            high_pool = self._split_high_confidence_pool(candidates)
            candidate_pool = self._split_candidate_pool(candidates)
            low_pool = self._split_low_confidence_pool(candidates)
            accepted_ids = [
                result.document_id for result in high_pool[: self.config.max_detection_pseudo_per_round]
            ]
            accepted_id_set = set(accepted_ids)
            round_pool_snapshots.append(
                {
                    "round_id": round_id,
                    "high_confidence_pool": self._render_detection_pool(documents_by_id, high_pool),
                    "candidate_pool": self._render_detection_pool(documents_by_id, candidate_pool),
                    "low_confidence_pool": self._render_detection_pool(documents_by_id, low_pool),
                }
            )
            if not accepted_id_set:
                print(f"[Step1] 第 {round_id} 轮没有新增高置信伪标签。", flush=True)
                break
            next_unlabeled = []
            for doc in unlabeled:
                result = next(item for item in candidates if item.document_id == doc.document_id)
                accepted = doc.document_id in accepted_id_set
                pool_name = self._resolve_detection_pool(result)
                logs.append(
                    PseudoLabelLog(
                        round_id=round_id,
                        stage="detection",
                        document_id=doc.document_id,
                        assigned_label="1" if result.is_event else "0",
                        confidence=result.confidence,
                        pool_name=pool_name,
                        accepted=accepted,
                    )
                )
                if accepted:
                    labeled[doc.document_id] = DetectionResult(
                        document_id=doc.document_id,
                        is_event=result.is_event,
                        has_new_fact=result.has_new_fact,
                        has_market_impact_path=result.has_market_impact_path,
                        confidence=result.confidence,
                        seed_category=result.seed_category,
                        reason=result.reason,
                        label_source="pseudo_label",
                    )
                    parked_high_results.pop(doc.document_id, None)
                elif pool_name == "candidate_pool":
                    next_unlabeled.append(doc)
                elif pool_name == "high_confidence_pool":
                    parked_high_results[doc.document_id] = result
                    next_unlabeled.append(doc)
                else:
                    parked_low_results[doc.document_id] = result
            unlabeled = next_unlabeled
            print(
                f"[Step1] 第 {round_id} 轮结束，高置信池 {len(high_pool)} 条，候选池 {len(candidate_pool)} 条，低置信池 {len(low_pool)} 条，新增吸收 {len(accepted_id_set)} 条。",
                flush=True,
            )
            self._emit_detection_progress(
                progress_callback=progress_callback,
                documents_by_id=documents_by_id,
                labeled=labeled,
                parked_high_results=parked_high_results,
                parked_low_results=parked_low_results,
                current_round_results=[],
                logs=logs,
                round_pool_snapshots=round_pool_snapshots,
                status="running",
                current_round_id=round_id,
            )

        examples = self._fewshot_detection_examples(documents_by_id, labeled)
        if unlabeled:
            print(f"[Step1] 开始最终补全识别，剩余 {len(unlabeled)} 条样本。", flush=True)
        for idx, doc in enumerate(unlabeled, 1):
            print(
                f"[Step1] 最终补全 {idx}/{len(unlabeled)}: {doc.document_id} | {doc.source_site} | {doc.title[:50]}",
                flush=True,
            )
            if doc.document_id in parked_high_results:
                labeled[doc.document_id] = parked_high_results[doc.document_id]
                self._emit_detection_progress(
                    progress_callback=progress_callback,
                    documents_by_id=documents_by_id,
                    labeled=labeled,
                    parked_high_results=parked_high_results,
                    parked_low_results=parked_low_results,
                    current_round_results=[],
                    logs=logs,
                    round_pool_snapshots=round_pool_snapshots,
                    status="running",
                    current_round_id=self.config.max_detection_rounds,
                )
                continue
            labeled[doc.document_id] = self.model.detect(doc, examples=examples)
            self._emit_detection_progress(
                progress_callback=progress_callback,
                documents_by_id=documents_by_id,
                labeled=labeled,
                parked_high_results=parked_high_results,
                parked_low_results=parked_low_results,
                current_round_results=[],
                logs=logs,
                round_pool_snapshots=round_pool_snapshots,
                status="running",
                current_round_id=self.config.max_detection_rounds,
            )
        for document_id, result in parked_low_results.items():
            labeled.setdefault(document_id, result)
        print("[Step1] 识别完成。", flush=True)
        final_high_confidence_pool = [
            self._result_with_document(doc, parked_high_results[doc.document_id])
            for doc in unlabeled
            if doc.document_id in parked_high_results
        ]
        final_candidate_pool = [
            self._result_with_document(doc, labeled[doc.document_id])
            for doc in unlabeled
            if doc.document_id not in parked_high_results
        ]
        final_low_confidence_pool = [
            self._result_with_document(documents_by_id[document_id], result)
            for document_id, result in parked_low_results.items()
        ]
        final_state = {
            "round_pool_snapshots": round_pool_snapshots,
            "final_high_confidence_pool": final_high_confidence_pool,
            "final_candidate_pool": final_candidate_pool,
            "final_low_confidence_pool": final_low_confidence_pool,
        }
        self._emit_detection_progress(
            progress_callback=progress_callback,
            documents_by_id=documents_by_id,
            labeled=labeled,
            parked_high_results=parked_high_results,
            parked_low_results=parked_low_results,
            current_round_results=[],
            logs=logs,
            round_pool_snapshots=round_pool_snapshots,
            status="completed",
            current_round_id=self.config.max_detection_rounds,
            final_state=final_state,
        )
        return labeled, logs, final_state

    def _emit_detection_progress(
        self,
        progress_callback: Optional[Callable[[Dict[str, DetectionResult], List[PseudoLabelLog], Dict[str, Any]], None]],
        documents_by_id: Dict[str, PreparedDocument],
        labeled: Dict[str, DetectionResult],
        parked_high_results: Dict[str, DetectionResult],
        parked_low_results: Dict[str, DetectionResult],
        current_round_results: List[DetectionResult],
        logs: List[PseudoLabelLog],
        round_pool_snapshots: List[Dict[str, Any]],
        status: str,
        current_round_id: int,
        final_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        if progress_callback is None:
            return
        current_results: Dict[str, DetectionResult] = {}
        current_results.update(labeled)
        current_results.update(parked_high_results)
        current_results.update(parked_low_results)
        for result in current_round_results:
            current_results[result.document_id] = result
        pool_state = final_state or {
            "round_pool_snapshots": round_pool_snapshots,
            "current_round_id": current_round_id,
            "current_round_results": self._render_detection_pool(documents_by_id, current_round_results),
            "parked_high_confidence_pool": self._render_detection_pool(
                documents_by_id,
                list(parked_high_results.values()),
            ),
            "parked_low_confidence_pool": self._render_detection_pool(
                documents_by_id,
                list(parked_low_results.values()),
            ),
            "final_high_confidence_pool": [],
            "final_candidate_pool": [],
            "final_low_confidence_pool": [],
        }
        pool_state["status"] = status
        progress_callback(current_results, list(logs), pool_state)

    def _resolve_detection_pool(self, result: DetectionResult) -> str:
        if result.is_event and result.confidence >= self.config.detection_high_positive_threshold:
            return "high_confidence_pool"
        if (not result.is_event) and result.confidence <= self.config.detection_high_negative_threshold:
            return "high_confidence_pool"
        if result.is_event and result.confidence >= self.config.detection_candidate_positive_threshold:
            return "candidate_pool"
        if (not result.is_event) and result.confidence <= self.config.detection_candidate_negative_threshold:
            return "candidate_pool"
        return "low_confidence_pool"

    def _split_high_confidence_pool(self, results: List[DetectionResult]) -> List[DetectionResult]:
        items = [item for item in results if self._resolve_detection_pool(item) == "high_confidence_pool"]
        return sorted(
            items,
            key=lambda item: item.confidence if item.is_event else (1.0 - item.confidence),
            reverse=True,
        )

    def _split_candidate_pool(self, results: List[DetectionResult]) -> List[DetectionResult]:
        items = [item for item in results if self._resolve_detection_pool(item) == "candidate_pool"]
        return sorted(
            items,
            key=lambda item: abs(item.confidence - 0.5),
            reverse=True,
        )

    def _split_low_confidence_pool(self, results: List[DetectionResult]) -> List[DetectionResult]:
        return [item for item in results if self._resolve_detection_pool(item) == "low_confidence_pool"]

    def run_classification(
        self,
        documents: List[PreparedDocument],
        detection_results: Dict[str, DetectionResult],
        existing_results: Optional[Dict[str, ClassificationResult]] = None,
        progress_callback: Optional[Callable[[Dict[str, ClassificationResult], List[PseudoLabelLog]], None]] = None,
    ) -> Tuple[Dict[str, ClassificationResult], List[PseudoLabelLog]]:
        existing_results = existing_results or {}
        event_docs = [doc for doc in documents if detection_results[doc.document_id].is_event]
        documents_by_id = {doc.document_id: doc for doc in event_docs}
        labeled: Dict[str, ClassificationResult] = {}
        unlabeled: List[PreparedDocument] = []
        logs: List[PseudoLabelLog] = []

        for doc in event_docs:
            if doc.document_id in existing_results:
                labeled[doc.document_id] = existing_results[doc.document_id]
            elif doc.event_type:
                labeled[doc.document_id] = ClassificationResult(
                    document_id=doc.document_id,
                    event_type=doc.event_type,
                    confidence=1.0,
                    reason="??????",
                    label_source="human_seed",
                    event_summary=doc.title,
                    duration_type="???",
                    heat=0.3,
                    heat_signal="?",
                    heat_comment_count=None,
                    keyword="",
                    keyword_score=1,
                    scale_score=0.0,
                    event_intensity=0.33,
                    influence_range=0.3,
                    attribute_score=0.225,
                )
            else:
                unlabeled.append(doc)

        self._emit_classification_progress(progress_callback, labeled, logs)

        if not labeled:
            for doc in event_docs:
                labeled[doc.document_id] = self.model.classify(doc, examples=[])
                self._emit_classification_progress(progress_callback, labeled, logs)
            return labeled, logs

        for round_id in range(1, self.config.max_classification_rounds + 1):
            if not unlabeled:
                break
            examples = self._fewshot_classification_examples(documents_by_id, labeled)
            candidates: List[ClassificationResult] = []
            for doc in unlabeled:
                candidates.append(self.model.classify(doc, examples=examples))
                partial = {result.document_id: result for result in candidates}
                self._emit_classification_progress(progress_callback, {**labeled, **partial}, logs)
            accepted_map: Dict[str, List[ClassificationResult]] = {}
            for result in candidates:
                if result.confidence >= self.config.classification_confidence_threshold:
                    accepted_map.setdefault(result.event_type, []).append(result)
            accepted_id_set = set()
            for _, items in accepted_map.items():
                items = sorted(items, key=lambda item: item.confidence, reverse=True)
                for item in items[: self.config.max_classification_pseudo_per_class]:
                    accepted_id_set.add(item.document_id)
            if not accepted_id_set:
                break

            next_unlabeled = []
            for doc in unlabeled:
                result = next(item for item in candidates if item.document_id == doc.document_id)
                accepted = doc.document_id in accepted_id_set
                logs.append(
                    PseudoLabelLog(
                        round_id=round_id,
                        stage="classification",
                        document_id=doc.document_id,
                        assigned_label=result.event_type,
                        confidence=result.confidence,
                        pool_name="classification_pool",
                        accepted=accepted,
                    )
                )
                if accepted:
                    labeled[doc.document_id] = ClassificationResult(
                        document_id=doc.document_id,
                        event_type=result.event_type,
                        confidence=result.confidence,
                        reason=result.reason,
                        label_source="pseudo_label",
                        event_summary=result.event_summary,
                        duration_type=result.duration_type,
                        heat=result.heat,
                        heat_signal=result.heat_signal,
                        heat_comment_count=result.heat_comment_count,
                        keyword=result.keyword,
                        keyword_score=result.keyword_score,
                        scale_score=result.scale_score,
                        event_intensity=result.event_intensity,
                        influence_range=result.influence_range,
                        attribute_score=result.attribute_score,
                    )
                else:
                    next_unlabeled.append(doc)
            unlabeled = next_unlabeled
            self._emit_classification_progress(progress_callback, labeled, logs)

        examples = self._fewshot_classification_examples(documents_by_id, labeled)
        for doc in unlabeled:
            labeled[doc.document_id] = self.model.classify(doc, examples=examples)
            self._emit_classification_progress(progress_callback, labeled, logs)
        return labeled, logs

    @staticmethod
    def _emit_classification_progress(
        progress_callback: Optional[Callable[[Dict[str, ClassificationResult], List[PseudoLabelLog]], None]],
        labeled: Dict[str, ClassificationResult],
        logs: List[PseudoLabelLog],
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(dict(labeled), list(logs))

    @staticmethod
    def _load_existing_classification_state(output_dir: str) -> Tuple[Dict[str, ClassificationResult], List[PseudoLabelLog]]:
        output = Path(output_dir)
        results_path = output / "classification_results.csv"
        logs_path = output / "pseudo_label_log.csv"
        existing_results: Dict[str, ClassificationResult] = {}
        existing_logs: List[PseudoLabelLog] = []

        if results_path.exists():
            try:
                df = pd.read_csv(results_path).fillna("")
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            for _, row in df.iterrows():
                document_id = str(row.get("document_id") or "").strip()
                event_type = str(row.get("event_type") or "").strip()
                if not document_id or not event_type:
                    continue
                existing_results[document_id] = ClassificationResult(
                    document_id=document_id,
                    event_type=event_type,
                    confidence=float(row.get("confidence") or 0.0),
                    reason=str(row.get("reason") or ""),
                    label_source=str(row.get("label_source") or "llm"),
                    event_summary=str(row.get("event_summary") or row.get("summary") or ""),
                    duration_type=str(row.get("duration_type") or ""),
                    heat=float(row.get("heat") or 0.0),
                    heat_signal=str(row.get("heat_signal") or ""),
                    heat_comment_count=(int(row.get("heat_comment_count")) if str(row.get("heat_comment_count") or "").strip() else None),
                    keyword=str(row.get("keyword") or ""),
                    keyword_score=int(row.get("keyword_score") or 1),
                    scale_score=float(row.get("scale_score") or 0.0),
                    event_intensity=float(row.get("event_intensity") or 0.0),
                    influence_range=float(row.get("influence_range") or 0.3),
                    attribute_score=float(row.get("attribute_score") or 0.0),
                )

        if logs_path.exists():
            try:
                df = pd.read_csv(logs_path).fillna("")
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            for _, row in df.iterrows():
                if str(row.get("stage") or "") != "classification":
                    continue
                existing_logs.append(
                    PseudoLabelLog(
                        round_id=int(row.get("round_id") or 0),
                        stage="classification",
                        document_id=str(row.get("document_id") or ""),
                        assigned_label=str(row.get("assigned_label") or ""),
                        confidence=float(row.get("confidence") or 0.0),
                        pool_name=str(row.get("pool_name") or ""),
                        accepted=str(row.get("accepted")).lower() in {"true", "1"},
                    )
                )

        return existing_results, existing_logs

    def _export_classification_progress(
        self,
        output_dir: str,
        documents: Iterable[PreparedDocument],
        detection_results: Dict[str, DetectionResult],
        classification_results: Dict[str, ClassificationResult],
        detection_logs: Iterable[PseudoLabelLog],
        classification_logs: Iterable[PseudoLabelLog],
    ) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        document_list = list(documents)
        documents_by_id = {doc.document_id: doc for doc in document_list}
        pd.DataFrame([doc.to_dict() for doc in document_list]).to_csv(
            output / "prepared_documents.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame([result.to_dict() for result in detection_results.values()]).to_csv(
            output / "detection_results.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame([result.to_dict() for result in classification_results.values()]).to_csv(
            output / "classification_results.csv", index=False, encoding="utf-8-sig"
        )
        all_logs = list(detection_logs) + list(classification_logs)
        pd.DataFrame([log.to_dict() for log in all_logs]).to_csv(
            output / "pseudo_label_log.csv", index=False, encoding="utf-8-sig"
        )
        classification_rows = [
            self._classification_result_with_document(
                documents_by_id[result.document_id],
                detection_results[result.document_id],
                result,
            )
            for result in classification_results.values()
            if result.document_id in documents_by_id and result.document_id in detection_results
        ]
        (output / "classification_results.json").write_text(
            json.dumps(classification_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (output / "classification_results.jsonl").open("w", encoding="utf-8") as fh:
            for row in classification_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def run_extraction(
        self,
        documents: List[PreparedDocument],
        detection_results: Dict[str, DetectionResult],
        classification_results: Dict[str, ClassificationResult],
    ) -> List[StructuredEvent]:
        events: List[StructuredEvent] = []
        for doc in documents:
            detection = detection_results[doc.document_id]
            if not detection.is_event:
                continue
            classification = classification_results[doc.document_id]
            events.extend(self.model.extract(doc, classification.event_type))
        return events

    def export(
        self,
        output_dir: str,
        documents: Iterable[PreparedDocument],
        detection_results: Dict[str, DetectionResult],
        classification_results: Dict[str, ClassificationResult],
        events: Iterable[StructuredEvent],
        logs: Iterable[PseudoLabelLog],
    ) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        document_list = list(documents)
        documents_by_id = {doc.document_id: doc for doc in document_list}
        document_rows = [doc.to_dict() for doc in document_list]
        event_rows = [event.to_dict() for event in events]
        log_rows = [log.to_dict() for log in logs]

        pd.DataFrame(document_rows).to_csv(output / "prepared_documents.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([result.to_dict() for result in detection_results.values()]).to_csv(
            output / "detection_results.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame([result.to_dict() for result in classification_results.values()]).to_csv(
            output / "classification_results.csv", index=False, encoding="utf-8-sig"
        )
        classification_rows = [
            self._classification_result_with_document(
                documents_by_id[result.document_id],
                detection_results[result.document_id],
                result,
            )
            for result in classification_results.values()
            if result.document_id in documents_by_id and result.document_id in detection_results
        ]
        (output / "classification_results.json").write_text(
            json.dumps(classification_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (output / "classification_results.jsonl").open("w", encoding="utf-8") as fh:
            for row in classification_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        pd.DataFrame(event_rows).to_csv(output / "structured_events.csv", index=False, encoding="utf-8-sig")
        with (output / "structured_events.jsonl").open("w", encoding="utf-8") as fh:
            for event in event_rows:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        pd.DataFrame(log_rows).to_csv(output / "pseudo_label_log.csv", index=False, encoding="utf-8-sig")

        summary = {
            "document_count": len(document_rows),
            "event_document_count": sum(1 for result in detection_results.values() if result.is_event),
            "structured_event_count": len(event_rows),
        }
        (output / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(self, input_path: str, output_dir: str) -> dict:
        documents = self.load_documents(input_path)
        detection_results, detection_logs, _ = self.run_detection(documents)
        existing_classification_results, existing_classification_logs = self._load_existing_classification_state(
            output_dir
        )

        if existing_classification_results:
            print(
                f"[Classification] 已恢复 {len(existing_classification_results)} 条已完成分类结果，将从断点继续。",
                flush=True,
            )

        def classification_progress_callback(
            current_results: Dict[str, ClassificationResult],
            current_logs: List[PseudoLabelLog],
        ) -> None:
            self._export_classification_progress(
                output_dir=output_dir,
                documents=documents,
                detection_results=detection_results,
                classification_results=current_results,
                detection_logs=detection_logs,
                classification_logs=existing_classification_logs + current_logs,
            )

        classification_results, classification_logs = self.run_classification(
            documents,
            detection_results,
            existing_results=existing_classification_results,
            progress_callback=classification_progress_callback,
        )
        events = self.run_extraction(documents, detection_results, classification_results)
        all_logs = detection_logs + existing_classification_logs + classification_logs
        self.export(
            output_dir=output_dir,
            documents=documents,
            detection_results=detection_results,
            classification_results=classification_results,
            events=events,
            logs=all_logs,
        )
        return {
            "documents": documents,
            "detection_results": detection_results,
            "classification_results": classification_results,
            "events": events,
            "logs": all_logs,
        }
