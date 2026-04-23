import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .llm import BaseTask2Model
from .schemas import CompanyProfile, CompanySeed, EventCompanyLink, Task2Config, Task2Event


def _to_float(value):
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


GENERIC_TERMS = {
    "行业", "市场", "公司", "业务", "产品", "服务", "增长", "下降", "影响", "提升", "利好", "利空",
    "需求", "供给", "价格", "政策", "事件", "发展", "领域", "布局", "平台", "项目", "方案",
}


class Task2Pipeline:
    def __init__(self, model: BaseTask2Model, config: Task2Config):
        self.model = model
        self.config = config

    @staticmethod
    def _load_json(path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _append_jsonl(path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _load_jsonl(path: Path) -> List[dict]:
        if not path.exists():
            return []
        rows = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                rows.append(json.loads(text))
        return rows

    def load_company_seeds(self, input_path: str) -> List[CompanySeed]:
        path = Path(input_path)
        payload = self._load_json(path)
        if not isinstance(payload, list):
            raise ValueError("Company input must be a JSON array.")
        seeds: List[CompanySeed] = []
        for idx, item in enumerate(payload, 1):
            if not isinstance(item, dict):
                continue
            stock_code = str(item.get("stock_code") or item.get("股票代码") or "").strip()
            stock_name = str(item.get("stock_name") or item.get("股票名称") or "").strip()
            industry_seed = str(item.get("industry_lv1_seed") or item.get("主要行业") or "").strip()
            order = int(item.get("order") or item.get("序号") or idx)
            if not stock_code or not stock_name:
                continue
            seeds.append(CompanySeed(order=order, stock_code=stock_code, stock_name=stock_name, industry_seed=industry_seed))
        if not seeds:
            raise ValueError(f"No valid companies found in {path}")
        return seeds

    @staticmethod
    def load_profiles(input_path: str) -> List[CompanyProfile]:
        path = Path(input_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload.get("companies", []) if isinstance(payload, dict) else payload
        profiles: List[CompanyProfile] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            profiles.append(CompanyProfile(**item))
        return profiles

    def load_events(self, input_path: str) -> List[Task2Event]:
        path = Path(input_path)
        payload = self._load_json(path)
        if not isinstance(payload, list):
            raise ValueError("Events input must be a JSON array.")
        events: List[Task2Event] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            event_type = str(item.get("event_type") or "").strip()
            sample_id = str(item.get("sample_id") or item.get("document_id") or "").strip()
            if not sample_id or not event_type:
                continue

            event_summary = str(item.get("event_summary") or item.get("event_attribute") or "").strip()
            impact_direction = str(item.get("impact_direction") or "").strip()
            if not impact_direction:
                attribute_score = _to_float(item.get("attribute_score"))
                if attribute_score >= 0.8:
                    impact_direction = "正向"
                elif 0 < attribute_score <= 0.25:
                    impact_direction = "负向"

            events.append(
                Task2Event(
                    sample_id=sample_id,
                    source_site=str(item.get("source_site") or "").strip(),
                    title=str(item.get("title") or "").strip(),
                    url=str(item.get("url") or "").strip(),
                    publish_time=str(item.get("publish_time") or "").strip(),
                    event_type=event_type,
                    event_summary=event_summary,
                    duration_type=str(item.get("duration_type") or "").strip(),
                    event_attribute=event_summary,
                    keyword=str(item.get("keyword") or "").strip(),
                    heat=_to_float(item.get("heat") or item.get("sentiment_heat")),
                    keyword_score=_to_float(item.get("keyword_score")),
                    scale_score=_to_float(item.get("scale_score") or item.get("materiality_score")),
                    event_intensity=_to_float(item.get("event_intensity")),
                    influence_range=_to_float(item.get("influence_range")),
                    attribute_score=_to_float(item.get("attribute_score")),
                    impact_direction=impact_direction,
                    correlation_logic=str(item.get("correlation_logic") or "").strip(),
                    classification_confidence=_to_float(item.get("classification_confidence")),
                    classification_reason=str(item.get("classification_reason") or "").strip(),
                )
            )
        if not events:
            raise ValueError(f"No valid events found in {path}")
        return events

    def normalize_company_seeds(self, seeds: Iterable[CompanySeed], output_path: str) -> List[CompanySeed]:
        seed_list = sorted(list(seeds), key=lambda item: (item.order, item.stock_code))
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([item.to_dict() for item in seed_list], ensure_ascii=False, indent=2), encoding="utf-8")
        return seed_list

    @staticmethod
    def _profile_progress_paths(output_dir: Path) -> Tuple[Path, Path, Path]:
        return (
            output_dir / "company_profiles.jsonl",
            output_dir / "company_profiles.json",
            output_dir / "company_profiles.csv",
        )

    @staticmethod
    def _link_progress_paths(output_dir: Path) -> Tuple[Path, Path, Path]:
        return (
            output_dir / "event_company_links.jsonl",
            output_dir / "event_company_links.json",
            output_dir / "event_company_links.csv",
        )

    def _export_profile_snapshot(self, output_dir: Path, ordered_codes: List[str], profile_map: Dict[str, CompanyProfile]) -> None:
        _, json_path, csv_path = self._profile_progress_paths(output_dir)
        profiles = [profile_map[code] for code in ordered_codes if code in profile_map]
        json_path.write_text(json.dumps({"companies": [item.to_dict() for item in profiles]}, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([item.to_dict() for item in profiles]).to_csv(csv_path, index=False, encoding="utf-8-sig")

    def _export_link_snapshot(self, output_dir: Path, ordered_ids: List[str], link_map: Dict[str, EventCompanyLink]) -> None:
        _, json_path, csv_path = self._link_progress_paths(output_dir)
        links = [link_map[item_id] for item_id in ordered_ids if item_id in link_map]
        json_path.write_text(json.dumps([item.to_dict() for item in links], ensure_ascii=False, indent=2), encoding="utf-8")

        flat_rows = []
        for result in links:
            if not result.matched_companies:
                flat_rows.append(
                    {
                        "sample_id": result.sample_id,
                        "title": result.title,
                        "event_type": result.event_type,
                        "source_site": result.source_site,
                        "event_impact_direction": result.impact_direction,
                        "stock_code": "",
                        "stock_name": "",
                        "company_impact_direction": "",
                        "match_type": "",
                        "match_score": 0.0,
                        "reason": "",
                        "evidence_keywords": "",
                    }
                )
                continue
            for match in result.matched_companies:
                flat_rows.append(
                    {
                        "sample_id": result.sample_id,
                        "title": result.title,
                        "event_type": result.event_type,
                        "source_site": result.source_site,
                        "event_impact_direction": result.impact_direction,
                        "stock_code": match.stock_code,
                        "stock_name": match.stock_name,
                        "company_impact_direction": match.impact_direction,
                        "match_type": match.match_type,
                        "match_score": match.match_score,
                        "reason": match.reason,
                        "evidence_keywords": "|".join(match.evidence_keywords),
                    }
                )
        pd.DataFrame(flat_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    def _load_existing_profile_progress(self, output_dir: Path) -> Dict[str, CompanyProfile]:
        jsonl_path, _, _ = self._profile_progress_paths(output_dir)
        rows = self._load_jsonl(jsonl_path)
        profile_map: Dict[str, CompanyProfile] = {}
        for row in rows:
            stock_code = str(row.get("stock_code") or "").strip()
            if not stock_code:
                continue
            profile_map[stock_code] = CompanyProfile(**row)
        return profile_map

    def _load_existing_link_progress(self, output_dir: Path) -> Dict[str, EventCompanyLink]:
        jsonl_path, _, _ = self._link_progress_paths(output_dir)
        rows = self._load_jsonl(jsonl_path)
        link_map: Dict[str, EventCompanyLink] = {}
        for row in rows:
            sample_id = str(row.get("sample_id") or "").strip()
            if not sample_id:
                continue
            matches = [
                {
                    "stock_code": item.get("stock_code", ""),
                    "stock_name": item.get("stock_name", ""),
                    "impact_direction": item.get("impact_direction", ""),
                    "match_type": item.get("match_type", ""),
                    "match_score": float(item.get("match_score", 0.0) or 0.0),
                    "reason": item.get("reason", ""),
                    "evidence_keywords": item.get("evidence_keywords", []),
                }
                for item in row.get("matched_companies", [])
                if isinstance(item, dict)
            ]
            link_map[sample_id] = EventCompanyLink(
                sample_id=sample_id,
                title=str(row.get("title") or ""),
                event_type=str(row.get("event_type") or ""),
                source_site=str(row.get("source_site") or ""),
                impact_direction=str(row.get("impact_direction") or ""),
                matched_companies=[type("_M", (), m)() for m in []],
            )
            from .schemas import EventCompanyMatch
            link_map[sample_id] = EventCompanyLink(
                sample_id=sample_id,
                title=str(row.get("title") or ""),
                event_type=str(row.get("event_type") or ""),
                source_site=str(row.get("source_site") or ""),
                impact_direction=str(row.get("impact_direction") or ""),
                matched_companies=[EventCompanyMatch(**item) for item in matches],
            )
        return link_map

    def build_company_profiles(self, seeds: Iterable[CompanySeed], output_dir: str) -> List[CompanyProfile]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        seed_list = list(seeds)
        if self.config.max_companies > 0:
            seed_list = seed_list[: self.config.max_companies]

        ordered_codes = [item.stock_code for item in seed_list]
        profile_map = self._load_existing_profile_progress(output)
        jsonl_path, _, _ = self._profile_progress_paths(output)
        processed_codes = set(profile_map.keys())

        if processed_codes:
            print(f"[Task2] 已恢复 {len(processed_codes)} 家已完成公司画像，将从断点继续。", flush=True)

        remaining = [item for item in seed_list if item.stock_code not in processed_codes]
        for idx, seed in enumerate(remaining, 1):
            print(f"[Task2] 正在生成公司画像 {idx}/{len(remaining)}: {seed.stock_code} | {seed.stock_name}", flush=True)
            try:
                profiles = self.model.build_company_profiles([seed])
            except Exception as exc:
                raise RuntimeError(f"Task2 profile failed for {seed.stock_code} {seed.stock_name}: {exc}") from exc
            if len(profiles) != 1:
                raise RuntimeError(f"Task2 profile failed for {seed.stock_code} {seed.stock_name}: expected exactly 1 profile, got {len(profiles)}")
            profile = profiles[0]
            profile_map[profile.stock_code] = profile
            raw_path = output / "company_profiles_raw" / f"{seed.stock_code}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            self._append_jsonl(jsonl_path, profile.to_dict())
            self._export_profile_snapshot(output, ordered_codes, profile_map)

        return [profile_map[code] for code in ordered_codes if code in profile_map]

    @staticmethod
    def _clean_term(term: str) -> str:
        return str(term or "").strip()

    def _iter_profile_terms(self, profile: CompanyProfile) -> List[str]:
        values = [
            profile.stock_name,
            profile.company_full_name,
            profile.industry_lv1,
            profile.industry_lv2,
            profile.business_desc,
            profile.summary_for_matching,
        ]
        values.extend(profile.concept_tags)
        values.extend(profile.main_products)
        values.extend(profile.product_keywords)
        values.extend(profile.application_scenarios)
        values.extend(profile.industry_chain_position)
        values.extend(profile.upstream_keywords)
        values.extend(profile.downstream_keywords)
        values.extend(profile.event_sensitive_keywords)
        values.extend(profile.relation_entities)
        values.extend(profile.direct_match_aliases)
        values.extend(profile.industry_nodes)
        values.extend(profile.event_match_keywords)
        values.extend(profile.benefit_direction_keywords)
        values.extend(profile.risk_direction_keywords)
        seen = set()
        terms = []
        for raw in values:
            term = self._clean_term(raw)
            if not term or len(term) < 2 or term in GENERIC_TERMS or term in seen:
                continue
            seen.add(term)
            terms.append(term)
        return terms

    @staticmethod
    def _event_text(event: Task2Event) -> str:
        return " ".join([event.title, event.event_summary, event.keyword, event.event_attribute, event.correlation_logic, event.classification_reason, event.source_site])

    def _score_profile(self, event: Task2Event, profile: CompanyProfile) -> Tuple[float, List[str], bool]:
        text = self._event_text(event)
        score = 0.0
        evidence: List[str] = []
        direct_hit = False

        aliases = list(dict.fromkeys(profile.direct_match_aliases + [profile.stock_name, profile.company_full_name]))
        for alias in aliases:
            alias = self._clean_term(alias)
            if alias and alias in text:
                direct_hit = True
                score += 4.0 if alias == profile.stock_name else 3.2
                evidence.append(alias)

        if event.event_type in profile.event_sensitive_types:
            score += 0.35
        if profile.match_priority == "高":
            score += 0.25
        elif profile.match_priority == "中":
            score += 0.1

        industry_terms = [self._clean_term(profile.industry_lv1), self._clean_term(profile.industry_lv2)]
        for term in industry_terms:
            if term and term not in GENERIC_TERMS and term in text:
                score += 0.7
                evidence.append(term)

        for term in self._iter_profile_terms(profile):
            if term in aliases or term in industry_terms:
                continue
            if term in text:
                score += 0.22
                evidence.append(term)

        direction_terms = profile.benefit_direction_keywords if event.impact_direction == "正向" else profile.risk_direction_keywords
        for term in direction_terms:
            term = self._clean_term(term)
            if term and term not in GENERIC_TERMS and term in text:
                score += 0.25
                evidence.append(term)

        if event.influence_scope == "个股" and not direct_hit:
            score *= 0.55
        elif event.influence_scope == "全市场" and not direct_hit and len(evidence) < 2:
            score *= 0.75

        return round(score, 3), sorted(set(evidence))[:8], direct_hit

    def _recall_candidates(self, event: Task2Event, profiles: Iterable[CompanyProfile]) -> List[CompanyProfile]:
        scored: List[Tuple[float, bool, CompanyProfile]] = []
        for profile in profiles:
            score, _, direct_hit = self._score_profile(event, profile)
            threshold = self.config.min_direct_score if direct_hit else self.config.min_recall_score
            if score >= threshold:
                scored.append((score, direct_hit, profile))

        scored.sort(key=lambda item: (item[1], item[0]), reverse=True)
        return [item[2] for item in scored[: self.config.match_candidate_limit]]

    def match_events_to_companies(self, events: Iterable[Task2Event], profiles: Iterable[CompanyProfile], output_dir: str) -> List[EventCompanyLink]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        profile_list = list(profiles)
        event_list = list(events)
        if self.config.max_events > 0:
            event_list = event_list[: self.config.max_events]

        ordered_ids = [event.sample_id for event in event_list]
        link_map = self._load_existing_link_progress(output)
        jsonl_path, _, _ = self._link_progress_paths(output)
        processed_ids = set(link_map.keys())

        if processed_ids:
            print(f"[Task2] 已恢复 {len(processed_ids)} 条已完成事件匹配，将从断点继续。", flush=True)

        remaining = [event for event in event_list if event.sample_id not in processed_ids]
        for idx, event in enumerate(remaining, 1):
            candidates = self._recall_candidates(event, profile_list)
            print(f"[Task2] 正在匹配事件 {idx}/{len(remaining)}: {event.sample_id} | 候选公司 {len(candidates)} 家。", flush=True)
            if not candidates:
                result = EventCompanyLink(
                    sample_id=event.sample_id,
                    title=event.title,
                    event_type=event.event_type,
                    source_site=event.source_site,
                    impact_direction=event.impact_direction,
                    matched_companies=[],
                )
            else:
                try:
                    result = self.model.match_event_to_companies(
                        event=event,
                        candidate_profiles=candidates,
                        max_match_results=self.config.max_match_results,
                    )
                except Exception as exc:
                    raise RuntimeError(f"Task2 match failed for {event.sample_id} {event.title[:60]}: {exc}") from exc
            link_map[result.sample_id] = result
            self._append_jsonl(jsonl_path, result.to_dict())
            self._export_link_snapshot(output, ordered_ids, link_map)

        summary = {
            "event_count": len([item for item in ordered_ids if item in link_map]),
            "matched_event_count": sum(1 for item_id in ordered_ids if item_id in link_map and link_map[item_id].matched_companies),
            "total_links": sum(len(link_map[item_id].matched_companies) for item_id in ordered_ids if item_id in link_map),
        }
        (output / "task2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return [link_map[item_id] for item_id in ordered_ids if item_id in link_map]

    def run(self, company_input: str, events_input: str, output_dir: str, stage: str) -> dict:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        result = {}
        profiles: List[CompanyProfile] = []

        if stage in {"profile", "all"}:
            seeds = self.load_company_seeds(company_input)
            normalized = self.normalize_company_seeds(seeds, str(output / "normalized_hs300.json"))
            profiles = self.build_company_profiles(normalized, str(output))
            result["company_seeds"] = normalized
            result["company_profiles"] = profiles

        if stage in {"match", "all"}:
            if not profiles:
                profiles = self.load_profiles(str(output / "company_profiles.json"))
            events = self.load_events(events_input)
            links = self.match_events_to_companies(events, profiles, str(output))
            result["events"] = events
            result["links"] = links

        return result
