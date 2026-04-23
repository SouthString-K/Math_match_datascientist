from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


EVENT_TYPE_OPTIONS = ("政策类", "宏观类", "行业类", "公司类", "地缘类")
MATCH_TYPE_OPTIONS = ("直接相关", "间接受益", "潜在关联")


@dataclass
class CompanySeed:
    order: int
    stock_code: str
    stock_name: str
    industry_seed: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompanyProfile:
    stock_code: str
    stock_name: str
    company_full_name: str = ""
    industry_lv1: str = ""
    industry_lv2: str = ""
    concept_tags: List[str] = field(default_factory=list)
    business_desc: str = ""
    main_products: List[str] = field(default_factory=list)
    product_keywords: List[str] = field(default_factory=list)
    application_scenarios: List[str] = field(default_factory=list)
    industry_chain_position: List[str] = field(default_factory=list)
    upstream_keywords: List[str] = field(default_factory=list)
    downstream_keywords: List[str] = field(default_factory=list)
    event_sensitive_types: List[str] = field(default_factory=list)
    event_sensitive_keywords: List[str] = field(default_factory=list)
    possible_benefit_events: List[str] = field(default_factory=list)
    possible_risk_events: List[str] = field(default_factory=list)
    relation_entities: List[str] = field(default_factory=list)
    direct_match_aliases: List[str] = field(default_factory=list)
    industry_nodes: List[str] = field(default_factory=list)
    event_match_keywords: List[str] = field(default_factory=list)
    benefit_direction_keywords: List[str] = field(default_factory=list)
    risk_direction_keywords: List[str] = field(default_factory=list)
    match_priority: str = ""
    summary_for_matching: str = ""
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Task2Event:
    sample_id: str
    source_site: str
    title: str
    url: str
    publish_time: str
    event_type: str
    event_summary: str = ""
    duration_type: str = ""
    event_attribute: str = ""
    keyword: str = ""
    heat: float = 0.0
    keyword_score: float = 0.0
    scale_score: float = 0.0
    event_intensity: float = 0.0
    influence_range: float = 0.0
    attribute_score: float = 0.0
    impact_direction: str = ""
    correlation_logic: str = ""
    classification_confidence: float = 0.0
    classification_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventCompanyMatch:
    stock_code: str
    stock_name: str
    impact_direction: str = ""
    match_type: str = ""
    match_score: float = 0.0
    reason: str = ""
    evidence_keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventCompanyLink:
    sample_id: str
    title: str
    event_type: str
    source_site: str
    impact_direction: str
    matched_companies: List[EventCompanyMatch] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "title": self.title,
            "event_type": self.event_type,
            "source_site": self.source_site,
            "impact_direction": self.impact_direction,
            "matched_companies": [item.to_dict() for item in self.matched_companies],
        }


@dataclass
class Task2Config:
    profile_batch_size: int = 1
    profile_max_retries: int = 5
    match_candidate_limit: int = 12
    max_match_results: int = 5
    min_recall_score: float = 1.0
    min_direct_score: float = 3.0
    max_companies: int = 0
    max_events: int = 0
    enable_thinking: bool = False
