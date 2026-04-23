from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PreparedDocument:
    document_id: str
    title: str
    content: str
    cleaned_content: str
    combined_text: str
    url: str = ""
    publish_time: str = ""
    source_site: str = ""
    event_label: Optional[int] = None
    event_type: str = ""
    has_new_fact: Optional[int] = None
    has_market_impact_path: Optional[int] = None
    detection_reason: str = ""
    detection_confidence: Optional[float] = None
    detection_seed_category: str = ""
    detection_label_source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionResult:
    document_id: str
    is_event: bool
    has_new_fact: int
    has_market_impact_path: int
    confidence: float
    seed_category: str
    reason: str
    label_source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClassificationResult:
    document_id: str
    event_type: str
    confidence: float
    reason: str
    label_source: str
    event_summary: str = ""
    duration_type: str = ""
    heat: float = 0.0
    heat_signal: str = ""
    heat_comment_count: Optional[int] = None
    keyword: str = ""
    keyword_score: int = 1
    scale_score: float = 0.0
    event_intensity: float = 0.0
    influence_range: float = 0.3
    attribute_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredEvent:
    event_id: str
    document_id: str
    event_name: str
    event_time: str
    publish_time: str
    source_url: str
    source_site: str
    event_main_type: str
    event_tags: List[str] = field(default_factory=list)
    industry: str = ""
    impact_scope: str = ""
    event_intensity: int = 0
    sentiment_heat: int = 0
    significance: bool = False
    mentioned_entities: List[str] = field(default_factory=list)
    candidate_companies: List[str] = field(default_factory=list)
    summary: str = ""
    evidence_text: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["event_tags"] = "|".join(self.event_tags)
        data["mentioned_entities"] = "|".join(self.mentioned_entities)
        data["candidate_companies"] = "|".join(self.candidate_companies)
        return data


@dataclass
class PseudoLabelLog:
    round_id: int
    stage: str
    document_id: str
    assigned_label: str
    confidence: float
    pool_name: str
    accepted: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Task1Config:
    max_detection_rounds: int = 3
    max_classification_rounds: int = 3
    detection_confidence_threshold: float = 0.82
    detection_high_positive_threshold: float = 0.90
    detection_high_negative_threshold: float = 0.10
    detection_candidate_positive_threshold: float = 0.70
    detection_candidate_negative_threshold: float = 0.30
    classification_confidence_threshold: float = 0.80
    max_detection_pseudo_per_round: int = 25
    max_classification_pseudo_per_class: int = 4
    max_fewshot_examples: int = 12
    max_content_chars: int = 3000
    enable_thinking: bool = True
