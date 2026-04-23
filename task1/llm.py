import json
import os
import re
import time
from typing import Iterable, List, Optional

from openai import OpenAI

from . import heuristics
from .prompts import build_classification_input, build_detection_input, build_extraction_input
from .schemas import ClassificationResult, DetectionResult, PreparedDocument, StructuredEvent


def _extract_json_block(text: str):
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", text, re.S)
        if not match:
            raise
        return json.loads(match.group(1))


class BaseTask1Model:
    def detect(self, document: PreparedDocument, examples: Iterable[dict]) -> DetectionResult:
        raise NotImplementedError

    def classify(self, document: PreparedDocument, examples: Iterable[dict]) -> ClassificationResult:
        raise NotImplementedError

    def extract(self, document: PreparedDocument, event_type: str) -> List[StructuredEvent]:
        raise NotImplementedError


class MockTask1Model(BaseTask1Model):
    def detect(self, document: PreparedDocument, examples: Iterable[dict]) -> DetectionResult:
        return heuristics.detect_event(document)

    def classify(self, document: PreparedDocument, examples: Iterable[dict]) -> ClassificationResult:
        return heuristics.classify_event(document)

    def extract(self, document: PreparedDocument, event_type: str) -> List[StructuredEvent]:
        return heuristics.extract_events(document, event_type=event_type, confidence=0.72)


class DashScopeTask1Model(BaseTask1Model):
    VALID_SEED_CATEGORIES = {
        "high_confidence_positive",
        "high_confidence_negative",
        "medium_confidence_positive",
        "medium_confidence_negative",
        "boundary_positive",
        "boundary_negative",
    }
    EVENT_TYPE_ALIASES = {
        "政策类": "政策类",
        "政策": "政策类",
        "政策事件": "政策类",
        "宏观类": "宏观类",
        "宏观": "宏观类",
        "宏观事件": "宏观类",
        "行业类": "行业类",
        "行业": "行业类",
        "行业事件": "行业类",
        "公司类": "公司类",
        "公司": "公司类",
        "公司事件": "公司类",
        "企业类": "公司类",
        "企业事件": "公司类",
        "地缘类": "地缘类",
        "地缘": "地缘类",
        "地缘事件": "地缘类",
        "地缘政治": "地缘类",
        "地缘政治事件": "地缘类",
    }
    DURATION_ALIASES = {
        "脉冲型": "脉冲型",
        "短期型": "脉冲型",
        "短促型": "脉冲型",
        "中期型": "中期型",
        "中短期": "中期型",
        "长尾型": "长尾型",
        "长期型": "长尾型",
        "长期": "长尾型",
    }
    HEAT_SIGNAL_ALIASES = {
        "高": "高",
        "高热": "高",
        "高关注": "高",
        "high": "高",
        "中": "中",
        "中等": "中",
        "一般": "中",
        "mid": "中",
        "medium": "中",
        "低": "低",
        "低热": "低",
        "低关注": "低",
        "low": "低",
    }
    SUBJECT_WEIGHTS = {"公司类": 0.9, "行业类": 1.0, "政策类": 1.1, "宏观类": 1.1, "地缘类": 1.2}
    DURATION_WEIGHTS = {"脉冲型": 0.8, "中期型": 1.0, "长尾型": 1.2}
    HEAT_LEVEL_TO_VALUE = {"低": 0.3, "中": 0.6, "高": 1.0}
    KEYWORD_NORM = {1: 0.33, 2: 0.67, 3: 1.0}
    VALID_SCALE_SCORES = [0.0, 0.3, 0.6, 1.0]
    VALID_RANGE_SCORES = [0.3, 0.6, 1.0]
    DEFAULT_TOOLS = [
        {"type": "web_search"},
        {"type": "web_extractor"},
    ]

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
        enable_thinking: bool = False,
        max_retries: int = 5,
        retry_backoff_seconds: float = 2.0,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is missing.")
        self.base_url = base_url.rstrip("/")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self.model = model
        self.enable_thinking = enable_thinking
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def _build_tools(self) -> List[dict]:
        if self.enable_thinking:
            return list(self.DEFAULT_TOOLS)
        return [{"type": "web_search"}]

    @staticmethod
    def _response_error_message(response) -> str:
        error = getattr(response, "error", None)
        if error is None:
            return ""
        code = getattr(error, "code", "") or ""
        message = getattr(error, "message", "") or ""
        return f"{code}: {message}".strip(": ")

    @staticmethod
    def _is_retryable_error_message(message: str) -> bool:
        lowered = message.lower()
        # ?????????????????????????
        if any(
            key in lowered
            for key in [
                "inappropriate content",
                "inappropriate-content",
                "output data may contain inappropriate content",
            ]
        ):
            return False
        return any(
            key in lowered
            for key in [
                "too many requests",
                "throttled",
                "capacity limits",
                "internalerror.algo",
                "server_error",
                "503",
            ]
        )

    def _run(self, prompt: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    tools=self._build_tools(),
                    extra_body={
                        "enable_thinking": self.enable_thinking,
                    },
                )
                error_message = self._response_error_message(response)
                if error_message:
                    if self._is_retryable_error_message(error_message) and attempt < self.max_retries:
                        sleep_seconds = self.retry_backoff_seconds * attempt
                        print(
                            f"[LLM] 第 {attempt} 次调用遇到可重试错误，{sleep_seconds:.1f}s 后重试：{error_message}",
                            flush=True,
                        )
                        time.sleep(sleep_seconds)
                        continue
                    raise ValueError(f"Model response error: {error_message}")

                output_text = getattr(response, "output_text", None)
                if output_text:
                    return output_text

                reasoning_summaries: List[str] = []
                message_parts: List[str] = []
                for item in response.output:
                    if getattr(item, "type", None) == "reasoning":
                        for summary in getattr(item, "summary", []) or []:
                            text = getattr(summary, "text", None)
                            if text:
                                reasoning_summaries.append(text[:500])
                    elif getattr(item, "type", None) == "message":
                        for content in getattr(item, "content", []) or []:
                            text = getattr(content, "text", None)
                            if text:
                                message_parts.append(text)

                if message_parts:
                    return "\n".join(message_parts).strip()
                if reasoning_summaries:
                    raise ValueError(
                        "Model returned reasoning summaries but no final message: "
                        + " | ".join(reasoning_summaries[:2])
                    )
                raise ValueError(f"No message content found in model response: {response}")
            except Exception as exc:
                last_error = exc
                message = str(exc)
                if self._is_retryable_error_message(message) and attempt < self.max_retries:
                    sleep_seconds = self.retry_backoff_seconds * attempt
                    print(
                        f"[LLM] 第 {attempt} 次调用失败，{sleep_seconds:.1f}s 后重试：{message[:300]}",
                        flush=True,
                    )
                    time.sleep(sleep_seconds)
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unexpected empty retry loop in DashScopeTask1Model._run")

    @staticmethod
    def _round3(value: float) -> float:
        return round(float(value) + 1e-9, 3)

    @staticmethod
    def _parse_optional_int(value: object) -> Optional[int]:
        if value in (None, "", "null"):
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip()
        if not text:
            return None
        match = re.search(r"-?\d+", text.replace(",", ""))
        if not match:
            return None
        return int(match.group(0))

    @staticmethod
    def _parse_float(value: object, default: float = 0.0) -> float:
        if value in (None, "", "null"):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            text = str(value).strip()
            match = re.search(r"-?\d+(?:\.\d+)?", text)
            if match:
                return float(match.group(0))
            return default

    @classmethod
    def _snap_score(cls, value: object, allowed: List[float], field_name: str, document_id: str) -> float:
        score = cls._parse_float(value, default=allowed[0])
        nearest = min(allowed, key=lambda item: abs(item - score))
        if abs(nearest - score) > 0.051:
            raise ValueError(f"Classification response invalid {field_name} for {document_id}: {value}")
        return cls._round3(nearest)

    @classmethod
    def _normalize_seed_category(cls, raw_value: str, is_event: bool, confidence: float) -> str:
        value = str(raw_value or "").strip()
        if value in cls.VALID_SEED_CATEGORIES:
            if is_event and value.endswith("negative"):
                return cls._category_from_confidence(is_event, confidence)
            if (not is_event) and value.endswith("positive"):
                return cls._category_from_confidence(is_event, confidence)
            return value
        return cls._category_from_confidence(is_event, confidence)

    @staticmethod
    def _category_from_confidence(is_event: bool, confidence: float) -> str:
        if is_event:
            if confidence >= 0.90:
                return "high_confidence_positive"
            if confidence >= 0.70:
                return "medium_confidence_positive"
            return "boundary_positive"
        if confidence <= 0.10:
            return "high_confidence_negative"
        if confidence <= 0.30:
            return "medium_confidence_negative"
        return "boundary_negative"

    @classmethod
    def _normalize_event_type(cls, value: object) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if text in cls.EVENT_TYPE_ALIASES:
            return cls.EVENT_TYPE_ALIASES[text]
        compact = text.replace(" ", "")
        if compact in cls.EVENT_TYPE_ALIASES:
            return cls.EVENT_TYPE_ALIASES[compact]
        for key, normalized in cls.EVENT_TYPE_ALIASES.items():
            if key in compact:
                return normalized
        return ""

    @classmethod
    def _normalize_duration_type(cls, value: object) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if text in cls.DURATION_ALIASES:
            return cls.DURATION_ALIASES[text]
        compact = text.replace(" ", "")
        if compact in cls.DURATION_ALIASES:
            return cls.DURATION_ALIASES[compact]
        for key, normalized in cls.DURATION_ALIASES.items():
            if key in compact:
                return normalized
        return ""

    @classmethod
    def _normalize_heat_signal(cls, value: object) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        lowered = text.lower()
        if text in cls.HEAT_SIGNAL_ALIASES:
            return cls.HEAT_SIGNAL_ALIASES[text]
        if lowered in cls.HEAT_SIGNAL_ALIASES:
            return cls.HEAT_SIGNAL_ALIASES[lowered]
        for key, normalized in cls.HEAT_SIGNAL_ALIASES.items():
            if key in text or key in lowered:
                return normalized
        return ""

    @classmethod
    def _compute_heat(cls, comment_count: Optional[int], heat_signal: str, event_type: str) -> float:
        if comment_count is not None:
            if comment_count > 3000:
                heat = 1.0
            elif comment_count >= 500:
                heat = 0.6
            else:
                heat = 0.3
        else:
            heat = cls.HEAT_LEVEL_TO_VALUE.get(heat_signal, 0.3)
        if event_type in {"政策类", "宏观类", "地缘类"}:
            if heat <= 0.3:
                heat = 0.6
            elif heat <= 0.6:
                heat = 1.0
        return cls._round3(min(heat, 1.0))

    @classmethod
    def _compute_intensity(cls, keyword_score: int, scale_score: float) -> float:
        keyword_norm = cls.KEYWORD_NORM.get(keyword_score, 0.33)
        if scale_score > 0:
            return cls._round3(0.6 * keyword_norm + 0.4 * scale_score)
        return cls._round3(keyword_norm)

    @classmethod
    def _compute_attribute(cls, event_type: str, duration_type: str, heat: float, intensity: float, influence_range: float) -> float:
        base = 0.3 * heat + 0.4 * intensity + 0.3 * influence_range
        return cls._round3(base * cls.SUBJECT_WEIGHTS[event_type] * cls.DURATION_WEIGHTS[duration_type])

    def detect(self, document: PreparedDocument, examples: Iterable[dict]) -> DetectionResult:
        payload = _extract_json_block(self._run(build_detection_input(document, examples)))
        raw_is_event = payload.get("is_event", 0)
        if isinstance(raw_is_event, str):
            raw_is_event = raw_is_event.strip()
        is_event = int(raw_is_event) == 1 if isinstance(raw_is_event, (int, float, str)) else False

        raw_has_new_fact = payload.get("has_new_fact", 0)
        if isinstance(raw_has_new_fact, str):
            raw_has_new_fact = raw_has_new_fact.strip()
        has_new_fact = int(raw_has_new_fact) if isinstance(raw_has_new_fact, (int, float, str)) else 0

        raw_has_market_impact_path = payload.get("has_market_impact_path", 0)
        if isinstance(raw_has_market_impact_path, str):
            raw_has_market_impact_path = raw_has_market_impact_path.strip()
        has_market_impact_path = (
            int(raw_has_market_impact_path)
            if isinstance(raw_has_market_impact_path, (int, float, str))
            else 0
        )

        confidence = float(payload.get("final_confidence", payload.get("confidence", 0.0)))
        seed_category = self._normalize_seed_category(payload.get("seed_category", ""), is_event, confidence)
        return DetectionResult(
            document_id=document.document_id,
            is_event=is_event,
            has_new_fact=1 if has_new_fact else 0,
            has_market_impact_path=1 if has_market_impact_path else 0,
            confidence=confidence,
            seed_category=seed_category,
            reason=str(payload.get("reason", "")),
            label_source="llm",
        )

    def classify(self, document: PreparedDocument, examples: Iterable[dict]) -> ClassificationResult:
        try:
            payload = _extract_json_block(self._run(build_classification_input(document, examples)))
        except Exception as exc:
            raise ValueError(f"Classification failed for {document.document_id}: {exc}") from exc
        raw_event_type = payload.get("event_type") or payload.get("subject_e") or payload.get("category")
        event_type = self._normalize_event_type(raw_event_type)
        if not event_type:
            preview = json.dumps(payload, ensure_ascii=False)[:600]
            raise ValueError(
                f"Classification response missing/invalid event_type for {document.document_id}: {preview}"
            )

        raw_duration = payload.get("duration_type") or payload.get("duration_e")
        duration_type = self._normalize_duration_type(raw_duration)
        if not duration_type:
            preview = json.dumps(payload, ensure_ascii=False)[:600]
            raise ValueError(
                f"Classification response missing/invalid duration_type for {document.document_id}: {preview}"
            )

        comment_count = self._parse_optional_int(
            payload.get("heat_comment_count")
            or payload.get("comment_count")
            or payload.get("comment_num")
        )
        heat_signal = self._normalize_heat_signal(payload.get("heat_signal") or payload.get("heat_level"))
        if comment_count is None and not heat_signal:
            preview = json.dumps(payload, ensure_ascii=False)[:600]
            raise ValueError(
                f"Classification response missing heat basis for {document.document_id}: {preview}"
            )

        keyword_score_raw = payload.get("keyword_score") or payload.get("KeywordScore") or payload.get("keywordscore")
        keyword_score = int(self._parse_float(keyword_score_raw, default=1.0))
        if keyword_score not in {1, 2, 3}:
            raise ValueError(
                f"Classification response invalid keyword_score for {document.document_id}: {keyword_score_raw}"
            )

        scale_score = self._snap_score(
            payload.get("scale_score") or payload.get("ScaleScore") or payload.get("ScaleScoree") or 0.0,
            self.VALID_SCALE_SCORES,
            "scale_score",
            document.document_id,
        )
        influence_range = self._snap_score(
            payload.get("influence_range") or payload.get("range") or payload.get("Range") or 0.3,
            self.VALID_RANGE_SCORES,
            "influence_range",
            document.document_id,
        )

        event_summary = str(payload.get("event_summary") or payload.get("summary_e") or payload.get("summary") or document.title).strip()
        keyword = str(payload.get("keyword") or payload.get("core_keyword") or "").strip()
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0) or 0.0)))
        reason = str(payload.get("reason", "")).strip()

        heat = self._compute_heat(comment_count, heat_signal, event_type)
        intensity = self._compute_intensity(keyword_score, scale_score)
        attribute_score = self._compute_attribute(event_type, duration_type, heat, intensity, influence_range)

        return ClassificationResult(
            document_id=document.document_id,
            event_type=event_type,
            confidence=confidence,
            reason=reason,
            label_source="llm",
            event_summary=event_summary,
            duration_type=duration_type,
            heat=heat,
            heat_signal=heat_signal or ("中" if heat >= 0.6 else "低"),
            heat_comment_count=comment_count,
            keyword=keyword,
            keyword_score=keyword_score,
            scale_score=scale_score,
            event_intensity=intensity,
            influence_range=influence_range,
            attribute_score=attribute_score,
        )

    def extract(self, document: PreparedDocument, event_type: str) -> List[StructuredEvent]:
        payload = _extract_json_block(self._run(build_extraction_input(document, event_type)))
        if not isinstance(payload, list):
            payload = [payload]
        events: List[StructuredEvent] = []
        for idx, item in enumerate(payload, 1):
            events.append(
                StructuredEvent(
                    event_id=f"{document.document_id}-E{idx}",
                    document_id=document.document_id,
                    event_name=str(item.get("event_name", document.title[:80])),
                    event_time=str(item.get("event_time", document.publish_time)),
                    publish_time=document.publish_time,
                    source_url=document.url,
                    source_site=document.source_site,
                    event_main_type=str(item.get("event_main_type", event_type)),
                    event_tags=[str(tag) for tag in item.get("event_tags", [])],
                    industry=str(item.get("industry", "")),
                    impact_scope=str(item.get("impact_scope", "")),
                    event_intensity=int(item.get("event_intensity", 0) or 0),
                    sentiment_heat=int(item.get("sentiment_heat", 0) or 0),
                    significance=bool(item.get("significance", False)),
                    mentioned_entities=[str(v) for v in item.get("mentioned_entities", [])],
                    candidate_companies=[str(v) for v in item.get("candidate_companies", [])],
                    summary=str(item.get("summary", "")),
                    evidence_text=str(item.get("evidence_text", "")),
                    confidence=float(item.get("confidence", 0.0) or 0.0),
                )
            )
        if not events:
            raise ValueError(f"Extraction response empty for {document.document_id}")
        return events
