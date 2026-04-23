import re
from typing import List

from .preprocessing import first_non_empty_sentence
from .schemas import ClassificationResult, DetectionResult, PreparedDocument, StructuredEvent


POSITIVE_KEYWORDS = {
    "政策",
    "条例",
    "办法",
    "通知",
    "指导意见",
    "关税",
    "制裁",
    "突破",
    "订单",
    "中标",
    "并购",
    "重组",
    "定增",
    "减持",
    "增持",
    "业绩预告",
    "回购",
    "停产",
    "复产",
    "涨价",
    "降价",
    "融资",
    "军工",
    "新能源",
    "芯片",
    "出口",
    "战争",
    "地缘",
    "监管",
    "央行",
    "财政",
    "利率",
}

NEGATIVE_KEYWORDS = {
    "收盘",
    "复盘",
    "点评",
    "午评",
    "早评",
    "知识",
    "科普",
    "教程",
    "百科",
    "如何",
    "是什么",
}

TYPE_KEYWORDS = {
    "政策类": {"政策", "通知", "指导意见", "监管", "证监会", "国务院", "发改委", "财政", "央行", "利率"},
    "宏观类": {"通胀", "GDP", "PMI", "失业率", "加息", "降息", "经济", "宏观"},
    "行业类": {"行业", "产业链", "涨价", "供需", "技术突破", "军工", "新能源", "半导体", "芯片"},
    "公司类": {"公司", "公告", "中标", "并购", "重组", "业绩", "回购", "增持", "减持", "订单"},
    "地缘类": {"战争", "制裁", "冲突", "地缘", "出口管制", "关税"},
}

HIGH_KEYWORDS = ["战争", "制裁", "禁令", "冲突升级", "重组", "暴增", "突破", "重大政策", "大额订单"]
MEDIUM_KEYWORDS = ["发布", "合作", "签约", "推进", "落地", "增长", "扩产"]
LOW_KEYWORDS = ["拟", "计划", "预计", "关注", "讨论"]

SUBJECT_WEIGHTS = {"公司类": 0.9, "行业类": 1.0, "政策类": 1.1, "宏观类": 1.1, "地缘类": 1.2}
DURATION_WEIGHTS = {"脉冲型": 0.8, "中期型": 1.0, "长尾型": 1.2}
HEAT_LEVEL_TO_VALUE = {"低": 0.3, "中": 0.6, "高": 1.0}
KEYWORD_NORM = {1: 0.33, 2: 0.67, 3: 1.0}


def _score_keywords(text: str, keywords: set) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def _round3(value: float) -> float:
    return round(float(value) + 1e-9, 3)


def _compute_heat(heat_signal: str, event_type: str) -> float:
    heat = HEAT_LEVEL_TO_VALUE.get(heat_signal, 0.3)
    if event_type in {"政策类", "宏观类", "地缘类"}:
        if heat <= 0.3:
            heat = 0.6
        elif heat <= 0.6:
            heat = 1.0
    return _round3(min(heat, 1.0))


def _compute_intensity(keyword_score: int, scale_score: float) -> float:
    keyword_norm = KEYWORD_NORM.get(keyword_score, 0.33)
    if scale_score > 0:
        return _round3(0.6 * keyword_norm + 0.4 * scale_score)
    return _round3(keyword_norm)


def _compute_attribute(event_type: str, duration_type: str, heat: float, intensity: float, influence_range: float) -> float:
    base = 0.3 * heat + 0.4 * intensity + 0.3 * influence_range
    return _round3(base * SUBJECT_WEIGHTS[event_type] * DURATION_WEIGHTS[duration_type])


def _pick_keyword(text: str) -> tuple[str, int]:
    for keyword in HIGH_KEYWORDS:
        if keyword in text:
            return keyword, 3
    for keyword in MEDIUM_KEYWORDS:
        if keyword in text:
            return keyword, 2
    for keyword in LOW_KEYWORDS:
        if keyword in text:
            return keyword, 1
    return "", 1


def _detect_scale_score(text: str) -> float:
    if re.search(r"\d+(?:\.\d+)?\s*(亿元|亿|万亿元|亿美元|万台|万吨|%|亿元人民币)", text):
        if re.search(r"\d{2,}(?:\.\d+)?\s*亿元|\d+(?:\.\d+)?\s*万亿元", text):
            return 1.0
        return 0.6
    if re.search(r"\d+", text):
        return 0.3
    return 0.0


def _guess_duration(event_type: str, text: str) -> str:
    if event_type in {"政策类", "宏观类", "地缘类"} and any(token in text for token in {"长期", "持续", "制度", "趋势", "出口管制"}):
        return "长尾型"
    if event_type in {"政策类", "宏观类", "行业类"}:
        return "中期型"
    return "脉冲型"


def detect_event(document: PreparedDocument) -> DetectionResult:
    text = f"{document.title}\n{document.cleaned_content}"
    pos = _score_keywords(text, POSITIVE_KEYWORDS)
    neg = _score_keywords(text, NEGATIVE_KEYWORDS)
    score = pos - neg
    is_event = score > 0 or any(keyword in document.title for keyword in POSITIVE_KEYWORDS)
    confidence = 0.55 + min(max(abs(score), 1), 5) * 0.08
    confidence = max(0.51, min(confidence, 0.95))
    if not is_event:
        confidence = max(0.51, min(0.88, 0.52 + neg * 0.08))
    reason = "命中事件关键词并具有股价影响可能性" if is_event else "更像资讯评论或泛化内容"
    return DetectionResult(
        document_id=document.document_id,
        is_event=is_event,
        has_new_fact=1 if is_event else 0,
        has_market_impact_path=1 if is_event else 0,
        confidence=confidence,
        seed_category="high_confidence_positive" if is_event else "high_confidence_negative",
        reason=reason,
        label_source="heuristic",
    )


def classify_event(document: PreparedDocument) -> ClassificationResult:
    text = f"{document.title}\n{document.cleaned_content}"
    best_type = "行业类"
    best_score = -1
    for event_type, keywords in TYPE_KEYWORDS.items():
        score = _score_keywords(text, keywords)
        if score > best_score:
            best_type = event_type
            best_score = score

    keyword, keyword_score = _pick_keyword(text)
    scale_score = _detect_scale_score(text)
    duration_type = _guess_duration(best_type, text)
    influence_range = 1.0 if best_type in {"政策类", "宏观类", "地缘类"} else (0.6 if best_type == "行业类" else 0.3)
    heat_signal = "高" if best_type in {"政策类", "宏观类", "地缘类"} else "中"
    heat = _compute_heat(heat_signal, best_type)
    intensity = _compute_intensity(keyword_score, scale_score)
    attribute_score = _compute_attribute(best_type, duration_type, heat, intensity, influence_range)
    confidence = min(0.93, 0.60 + min(max(best_score, 1), 5) * 0.07)

    return ClassificationResult(
        document_id=document.document_id,
        event_type=best_type,
        confidence=confidence,
        reason="根据标题与正文关键词抽取驱动主体、持续周期和量化中间变量，再按固定公式得到最终指标。",
        label_source="heuristic",
        event_summary=first_non_empty_sentence(document.cleaned_content) or document.title,
        duration_type=duration_type,
        heat=heat,
        heat_signal=heat_signal,
        heat_comment_count=None,
        keyword=keyword,
        keyword_score=keyword_score,
        scale_score=scale_score,
        event_intensity=intensity,
        influence_range=influence_range,
        attribute_score=attribute_score,
    )


def _extract_entities(text: str) -> List[str]:
    matches = re.findall(r"[A-Z]{2,5}|[\u4e00-\u9fa5]{2,8}(?:公司|集团|股份|银行|证券|科技|能源|军工)", text)
    unique: List[str] = []
    for match in matches:
        if match not in unique:
            unique.append(match)
    return unique[:8]


def extract_events(document: PreparedDocument, event_type: str, confidence: float) -> List[StructuredEvent]:
    text = document.cleaned_content
    summary = first_non_empty_sentence(text)
    entities = _extract_entities(f"{document.title}\n{text}")
    return [
        StructuredEvent(
            event_id=f"{document.document_id}-E1",
            document_id=document.document_id,
            event_name=document.title[:80] or summary[:80],
            event_time=document.publish_time,
            publish_time=document.publish_time,
            source_url=document.url,
            source_site=document.source_site,
            event_main_type=event_type,
            event_tags=[],
            industry="",
            impact_scope="行业" if event_type in {"政策类", "行业类", "宏观类"} else "个股",
            event_intensity=6,
            sentiment_heat=6,
            significance=True,
            mentioned_entities=entities,
            candidate_companies=entities,
            summary=summary,
            evidence_text=summary,
            confidence=confidence,
        )
    ]
