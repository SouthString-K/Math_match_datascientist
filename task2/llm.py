import json
import os
import re
import time
from typing import Iterable, List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime for mock mode
    OpenAI = None

from .prompts import build_match_input, build_profile_input
from .schemas import (
    EVENT_TYPE_OPTIONS,
    MATCH_TYPE_OPTIONS,
    CompanyProfile,
    CompanySeed,
    EventCompanyLink,
    EventCompanyMatch,
    Task2Event,
)


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


def _ensure_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if "、" in text:
        return [part.strip() for part in text.split("、") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


class BaseTask2Model:
    def build_company_profiles(self, companies: Iterable[CompanySeed]) -> List[CompanyProfile]:
        raise NotImplementedError

    def match_event_to_companies(
        self,
        event: Task2Event,
        candidate_profiles: Iterable[CompanyProfile],
        max_match_results: int,
    ) -> EventCompanyLink:
        raise NotImplementedError


class MockTask2Model(BaseTask2Model):
    INDUSTRY_EVENT_MAP = {
        "半导体": ["行业类", "公司类", "政策类"],
        "光伏设备": ["行业类", "政策类"],
        "生物制品": ["行业类", "公司类", "政策类"],
        "银行II": ["宏观类", "政策类", "公司类"],
        "证券II": ["宏观类", "政策类", "公司类"],
        "保险II": ["宏观类", "政策类", "公司类"],
        "航运港口": ["地缘类", "行业类", "宏观类"],
        "油气开采II": ["地缘类", "行业类", "宏观类"],
        "炼化及贸易": ["地缘类", "行业类", "宏观类"],
        "煤炭开采": ["宏观类", "行业类"],
        "乘用车": ["行业类", "政策类", "公司类"],
    }

    def build_company_profiles(self, companies: Iterable[CompanySeed]) -> List[CompanyProfile]:
        profiles: List[CompanyProfile] = []
        for item in companies:
            event_types = self.INDUSTRY_EVENT_MAP.get(item.industry_seed, ["行业类", "公司类"])
            profiles.append(
                CompanyProfile(
                    stock_code=item.stock_code,
                    stock_name=item.stock_name,
                    company_full_name="",
                    industry_lv1=item.industry_seed,
                    industry_lv2=item.industry_seed,
                    concept_tags=[item.industry_seed] if item.industry_seed else [],
                    business_desc=item.industry_seed,
                    main_products=[item.industry_seed] if item.industry_seed else [],
                    product_keywords=[item.stock_name, item.industry_seed] if item.industry_seed else [item.stock_name],
                    event_sensitive_types=event_types,
                    event_sensitive_keywords=[item.industry_seed] if item.industry_seed else [],
                    relation_entities=[item.stock_name],
                    direct_match_aliases=[item.stock_name],
                    industry_nodes=[item.industry_seed] if item.industry_seed else [],
                    event_match_keywords=[item.stock_name, item.industry_seed] if item.industry_seed else [item.stock_name],
                    match_priority="中",
                    summary_for_matching=f"{item.stock_name}，所属{item.industry_seed}，关注行业政策、景气和公司经营变化。".strip("，"),
                    confidence=0.35,
                    warnings=["mock画像"],
                )
            )
        return profiles

    def match_event_to_companies(
        self,
        event: Task2Event,
        candidate_profiles: Iterable[CompanyProfile],
        max_match_results: int,
    ) -> EventCompanyLink:
        event_text = " ".join([event.title, event.event_attribute, event.correlation_logic, event.classification_reason])
        scored = []
        for profile in candidate_profiles:
            score = 0.0
            evidence = []
            for alias in profile.direct_match_aliases + [profile.stock_name]:
                if alias and alias in event_text:
                    score += 0.55
                    evidence.append(alias)
            if event.event_type and event.event_type in profile.event_sensitive_types:
                score += 0.2
            for word in profile.event_match_keywords + profile.event_sensitive_keywords:
                if word and word in event_text:
                    score += 0.08
                    evidence.append(word)
            if score <= 0:
                continue
            scored.append(
                EventCompanyMatch(
                    stock_code=profile.stock_code,
                    stock_name=profile.stock_name,
                    impact_direction=event.impact_direction or "中性",
                    match_type="直接相关" if any(alias in event_text for alias in profile.direct_match_aliases) else "潜在关联",
                    match_score=min(round(score, 2), 0.95),
                    reason="基于简称/关键词和事件类型的启发式匹配。",
                    evidence_keywords=sorted(set(evidence))[:6],
                )
            )
        scored = sorted(scored, key=lambda item: item.match_score, reverse=True)[:max_match_results]
        return EventCompanyLink(
            sample_id=event.sample_id,
            title=event.title,
            event_type=event.event_type,
            source_site=event.source_site,
            impact_direction=event.impact_direction,
            matched_companies=scored,
        )


class DashScopeTask2Model(BaseTask2Model):
    DEFAULT_TOOLS = [{"type": "web_search"}, {"type": "web_extractor"}]

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
        enable_thinking: bool = False,
        max_retries: int = 5,
        retry_backoff_seconds: float = 2.0,
    ):
        if OpenAI is None:
            raise ImportError("openai package is required for dashscope provider. Please install requirements.txt first.")
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is missing.")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url.rstrip("/"))
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
        return any(key in lowered for key in ["too many requests", "throttled", "capacity limits", "internalerror.algo", "server_error", "503"])

    def _run(self, prompt: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    tools=self._build_tools(),
                    extra_body={"enable_thinking": self.enable_thinking},
                )
                error_message = self._response_error_message(response)
                if error_message:
                    if self._is_retryable_error_message(error_message) and attempt < self.max_retries:
                        sleep_seconds = self.retry_backoff_seconds * attempt
                        print(f"[Task2 LLM] 第 {attempt} 次调用可重试错误，{sleep_seconds:.1f}s 后重试：{error_message}", flush=True)
                        time.sleep(sleep_seconds)
                        continue
                    raise ValueError(f"Model response error: {error_message}")

                output_text = getattr(response, "output_text", None)
                if output_text:
                    return output_text

                message_parts: List[str] = []
                for item in getattr(response, "output", []) or []:
                    if getattr(item, "type", None) == "message":
                        for content in getattr(item, "content", []) or []:
                            text = getattr(content, "text", None)
                            if text:
                                message_parts.append(text)
                if message_parts:
                    return "\n".join(message_parts).strip()
                raise ValueError(f"No message content found in model response: {response}")
            except Exception as exc:
                last_error = exc
                if self._is_retryable_error_message(str(exc)) and attempt < self.max_retries:
                    sleep_seconds = self.retry_backoff_seconds * attempt
                    print(f"[Task2 LLM] 第 {attempt} 次调用失败，{sleep_seconds:.1f}s 后重试：{str(exc)[:300]}", flush=True)
                    time.sleep(sleep_seconds)
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unexpected empty retry loop in DashScopeTask2Model._run")

    @staticmethod
    def _normalize_event_types(values) -> List[str]:
        normalized = []
        for value in _ensure_list(values):
            if value == "技术突破类":
                value = "行业类"
            if value in EVENT_TYPE_OPTIONS and value not in normalized:
                normalized.append(value)
        return normalized

    @staticmethod
    def _normalize_match_type(value: object) -> str:
        text = str(value or "").strip()
        if text in MATCH_TYPE_OPTIONS:
            return text
        if "直接" in text:
            return "直接相关"
        if "间接" in text:
            return "间接受益"
        return "潜在关联" if text else ""

    @staticmethod
    def _normalize_direction(value: object) -> str:
        text = str(value or "").strip()
        if text in {"正向", "负向", "中性"}:
            return text
        if "利好" in text or "正" in text:
            return "正向"
        if "利空" in text or "负" in text:
            return "负向"
        return "中性" if text else ""

    def _profile_from_payload(self, payload: dict, seed: CompanySeed) -> CompanyProfile:
        industry_lv1 = str(payload.get("industry_lv1") or seed.industry_seed or "").strip()
        warnings = _ensure_list(payload.get("warnings"))

        payload_stock_code = str(payload.get("stock_code") or "").strip()
        payload_stock_name = str(payload.get("stock_name") or "").strip()
        if payload_stock_code and payload_stock_code != seed.stock_code:
            warnings.append(f"?????stock_code={payload_stock_code}???{seed.stock_code}???????????")
        if payload_stock_name and payload_stock_name != seed.stock_name:
            warnings.append(f"?????stock_name={payload_stock_name}???{seed.stock_name}???????????")

        profile = CompanyProfile(
            stock_code=seed.stock_code,
            stock_name=seed.stock_name,
            company_full_name=str(payload.get("company_full_name") or "").strip(),
            industry_lv1=industry_lv1,
            industry_lv2=str(payload.get("industry_lv2") or seed.industry_seed or "").strip(),
            concept_tags=_ensure_list(payload.get("concept_tags")),
            business_desc=str(payload.get("business_desc") or "").strip(),
            main_products=_ensure_list(payload.get("main_products")),
            product_keywords=_ensure_list(payload.get("product_keywords")),
            application_scenarios=_ensure_list(payload.get("application_scenarios")),
            industry_chain_position=_ensure_list(payload.get("industry_chain_position")),
            upstream_keywords=_ensure_list(payload.get("upstream_keywords")),
            downstream_keywords=_ensure_list(payload.get("downstream_keywords")),
            event_sensitive_types=self._normalize_event_types(payload.get("event_sensitive_types")),
            event_sensitive_keywords=_ensure_list(payload.get("event_sensitive_keywords")),
            possible_benefit_events=_ensure_list(payload.get("possible_benefit_events")),
            possible_risk_events=_ensure_list(payload.get("possible_risk_events")),
            relation_entities=_ensure_list(payload.get("relation_entities")),
            direct_match_aliases=_ensure_list(payload.get("direct_match_aliases")) or [seed.stock_name],
            industry_nodes=_ensure_list(payload.get("industry_nodes")),
            event_match_keywords=_ensure_list(payload.get("event_match_keywords")),
            benefit_direction_keywords=_ensure_list(payload.get("benefit_direction_keywords")),
            risk_direction_keywords=_ensure_list(payload.get("risk_direction_keywords")),
            match_priority=str(payload.get("match_priority") or "").strip(),
            summary_for_matching=str(payload.get("summary_for_matching") or "").strip(),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            warnings=warnings,
        )
        if not profile.direct_match_aliases:
            profile.direct_match_aliases = [seed.stock_name]
        if not profile.event_sensitive_types:
            profile.event_sensitive_types = ["行业类", "公司类"]
        return profile

    def build_company_profiles(self, companies: Iterable[CompanySeed]) -> List[CompanyProfile]:
        company_list = list(companies)
        payload = _extract_json_block(self._run(build_profile_input(company_list)))
        items = payload.get("companies", []) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise ValueError(f"Task2 profile response has invalid structure: {payload}")
        if len(company_list) == 1 and len(items) != 1:
            raise ValueError(f"Task2 profile response must contain exactly 1 company, got {len(items)}")

        seed_map = {item.stock_code: item for item in company_list}
        results: List[CompanyProfile] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Task2 profile response item is not an object: {item}")
            stock_code = str(item.get("stock_code") or "").strip()
            seed = seed_map.get(stock_code)
            if seed is None:
                for candidate in company_list:
                    if str(item.get("stock_name") or "").strip() == candidate.stock_name:
                        seed = candidate
                        break
            if seed is None:
                raise ValueError(f"Task2 profile response company not found in input batch: {item}")
            results.append(self._profile_from_payload(item, seed))

        if len(results) != len(company_list):
            raise ValueError(f"Task2 profile response count mismatch: expected {len(company_list)}, got {len(results)}")
        return results

    def match_event_to_companies(self, event: Task2Event, candidate_profiles: Iterable[CompanyProfile], max_match_results: int) -> EventCompanyLink:
        profiles = list(candidate_profiles)
        payload = _extract_json_block(self._run(build_match_input(event, profiles, max_match_results)))
        if not isinstance(payload, dict):
            raise ValueError(f"Task2 match response has invalid structure: {payload}")
        matches_payload = payload.get("matched_companies", [])
        if not isinstance(matches_payload, list):
            raise ValueError(f"Task2 match response matched_companies must be a list: {payload}")

        matches: List[EventCompanyMatch] = []
        for item in matches_payload:
            if not isinstance(item, dict):
                raise ValueError(f"Task2 match item is not an object: {item}")
            stock_code = str(item.get("stock_code") or "").strip()
            stock_name = str(item.get("stock_name") or "").strip()
            if not stock_code and not stock_name:
                raise ValueError(f"Task2 match item missing stock identifiers: {item}")
            match_score = float(item.get("match_score", 0.0) or 0.0)
            if match_score < 0.45:
                continue
            matches.append(
                EventCompanyMatch(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    impact_direction=self._normalize_direction(item.get("impact_direction")),
                    match_type=self._normalize_match_type(item.get("match_type")),
                    match_score=match_score,
                    reason=str(item.get("reason") or "").strip(),
                    evidence_keywords=_ensure_list(item.get("evidence_keywords")),
                )
            )
        matches = sorted(matches, key=lambda item: item.match_score, reverse=True)[:max_match_results]
        return EventCompanyLink(
            sample_id=event.sample_id,
            title=event.title,
            event_type=event.event_type,
            source_site=event.source_site,
            impact_direction=event.impact_direction,
            matched_companies=matches,
        )
