import json
from typing import Iterable, List

from .schemas import CompanyProfile, CompanySeed, Task2Event


TASK2_SYSTEM_PROMPT = (
    "You are a financial research assistant for event-driven stock strategy research. "
    "You must use web search and web extraction when needed, keep output strictly valid JSON, "
    "and avoid fabricating unsupported facts."
)


TASK2_PROFILE_PROMPT_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
你是上市公司画像构建助手，服务于事件驱动型投资研究中的“事件-公司关联挖掘”。

【任务目标】
请根据输入的单家公司信息，为该公司生成结构化公司画像。

【强约束：禁止幻觉 / 禁止编造】
1. 你必须先调用网页搜索和网页抓取工具，尽量查阅公司官网、年报摘要、权威财经资料或交易所公开信息后再作答；
2. 输入仅提供股票代码、公司简称和行业线索，你只能基于公开可验证信息补全画像；
3. 若无法可靠判断，字段必须置空，不可根据常识、猜测或行业经验擅自补全；
4. 禁止虚构财务数字、公告、合同、股权比例、合作关系、客户名单、供应商名单；
5. 若简称存在歧义、资料不足或多个来源冲突，必须降低 confidence，并在 warnings 中明确说明。

【分类约束】
1. 只允许在以下事件类型中选择：政策类、宏观类、行业类、公司类、地缘类；
2. 若遇到“技术突破”相关情形，请归并到行业类或公司类，不要输出“技术突破类”；
3. event_sensitive_types 建议输出 1 到 3 个，不要无差别全填。

【关键词质量约束】
1. direct_match_aliases 只保留常用简称、核心产品别名、常见业务别称，不要机械重复；
2. event_match_keywords 必须是可用于事件召回的高辨识度短语，尽量具体；
3. benefit_direction_keywords 和 risk_direction_keywords 必须是触发词短语，例如“涨价”“补贴加码”“供给收缩”“出口受限”；
4. 禁止输出过于空泛的词，例如“行业”“市场”“业务”“产品”“公司”“增长”“利好”“利空”。

【字段要求】
每家公司必须输出以下字段，不得遗漏：
- stock_code
- stock_name
- company_full_name
- industry_lv1
- industry_lv2
- concept_tags
- business_desc
- main_products
- product_keywords
- application_scenarios
- industry_chain_position
- upstream_keywords
- downstream_keywords
- event_sensitive_types
- event_sensitive_keywords
- possible_benefit_events
- possible_risk_events
- relation_entities
- direct_match_aliases
- industry_nodes
- event_match_keywords
- benefit_direction_keywords
- risk_direction_keywords
- match_priority
- summary_for_matching
- confidence
- warnings

【质量约束】
1. match_priority 仅在以下情形给“高”：公司简称辨识度强、核心产品高度聚焦、事件传导路径清晰；
2. summary_for_matching 需覆盖行业、产品、应用、概念、产业链信息，100字以内；
3. 所有标签型字段尽量使用短语，不要写长句；
4. confidence 取值 0 到 1。

【输入公司】
{companies_json}

【输出要求】
只输出合法 JSON，不要输出 Markdown、解释或额外文字：
{{
  "companies": [
    {{
      "stock_code": "",
      "stock_name": "",
      "company_full_name": "",
      "industry_lv1": "",
      "industry_lv2": "",
      "concept_tags": [],
      "business_desc": "",
      "main_products": [],
      "product_keywords": [],
      "application_scenarios": [],
      "industry_chain_position": [],
      "upstream_keywords": [],
      "downstream_keywords": [],
      "event_sensitive_types": [],
      "event_sensitive_keywords": [],
      "possible_benefit_events": [],
      "possible_risk_events": [],
      "relation_entities": [],
      "direct_match_aliases": [],
      "industry_nodes": [],
      "event_match_keywords": [],
      "benefit_direction_keywords": [],
      "risk_direction_keywords": [],
      "match_priority": "",
      "summary_for_matching": "",
      "confidence": 0.0,
      "warnings": []
    }}
  ]
}}
<|im_end|>
<|im_start|>assistant
"""


TASK2_MATCH_PROMPT_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
你是事件-公司关联匹配助手。

【任务目标】
请根据给定事件及候选公司画像，判断哪些公司与该事件存在较强关联，并给出方向、类型、分数和理由。

【强约束：禁止幻觉 / 禁止乱配】
1. 关联判断必须以“事件事实 -> 传导路径 -> 公司业务/产品/产业链节点”三段链条为基础；
2. 不要因为公司热门、市值大、同属A股核心资产就强行关联；
3. 若事件是宏观或全市场事件，但没有明确行业、产品、公司或产业链线索，应返回空列表；
4. 若证据只支持行业主题，不足以支持单家公司，应减少输出数量或返回空列表；
5. 若无法说明该公司为何比同类大量公司更值得被匹配，就不要输出；
6. 禁止依据常识臆断公司受益或受损，必须基于输入事件文本和公司画像中的可验证线索。

【match_type 判定标准】
- 直接相关：事件直接点名该公司、其简称/别名/核心产品，或事件直接作用于该公司主营业务、公告事项、核心经营活动
- 间接受益：事件未直接点名公司，但通过产业链、供需、成本、政策、订单传导，会较明确影响该公司
- 潜在关联：存在主题或业务相关性，但证据相对弱，仅可作为候选补充，不可滥用

【何时必须返回空列表】
1. 事件只有宏观环境变化，没有清晰受益/受损行业线索；
2. 候选公司与事件只有“事件类型相同”这一层弱联系；
3. 证据关键词过于泛化，例如仅匹配到“行业”“公司”“市场”“增长”等空泛词；
4. 无法形成清晰的传导链条；
5. 所有候选公司的 match_score 都不足 0.45。

【impact_direction 说明】
只能填写：正向、负向、中性。
若事件方向本身不明确，应优先填中性，不要硬判。

【match_score 说明】
- 0.85-1.00：高把握，存在直接点名或非常清晰的主营/产业链传导
- 0.65-0.84：中高把握，产业链或业务映射较明确
- 0.45-0.64：弱关联，仅在证据仍有一定价值时保留
- 低于 0.45 不应输出

【evidence_keywords 要求】
1. 必须来自事件文本与公司画像中真实出现的词或短语；
2. 优先输出公司简称、产品、产业链节点、事件主题词；
3. 不要输出空泛词。

【输入事件】
{event_json}

【候选公司画像】
{candidate_profiles_json}

【输出要求】
只输出一个合法 JSON 对象，不要输出任何解释：
{{
  "sample_id": "{sample_id}",
  "title": "{title}",
  "event_type": "{event_type}",
  "matched_companies": [
    {{
      "stock_code": "",
      "stock_name": "",
      "impact_direction": "正向/负向/中性",
      "match_type": "直接相关/间接受益/潜在关联",
      "match_score": 0.0,
      "reason": "",
      "evidence_keywords": []
    }}
  ]
}}
如果没有足够证据，请输出 matched_companies: []。
<|im_end|>
<|im_start|>assistant
"""


def build_profile_input(companies: Iterable[CompanySeed]) -> str:
    payload = [
        {
            "stock_code": item.stock_code,
            "stock_name": item.stock_name,
            "industry_seed": item.industry_seed,
        }
        for item in companies
    ]
    return TASK2_PROFILE_PROMPT_TEMPLATE.format(
        system=TASK2_SYSTEM_PROMPT,
        companies_json=json.dumps(payload, ensure_ascii=False, indent=2),
    )


def _candidate_profile_payload(profiles: Iterable[CompanyProfile]) -> List[dict]:
    payload = []
    for item in profiles:
        payload.append(
            {
                "stock_code": item.stock_code,
                "stock_name": item.stock_name,
                "industry_lv1": item.industry_lv1,
                "industry_lv2": item.industry_lv2,
                "concept_tags": item.concept_tags,
                "business_desc": item.business_desc,
                "main_products": item.main_products,
                "product_keywords": item.product_keywords,
                "application_scenarios": item.application_scenarios,
                "industry_chain_position": item.industry_chain_position,
                "upstream_keywords": item.upstream_keywords,
                "downstream_keywords": item.downstream_keywords,
                "event_sensitive_types": item.event_sensitive_types,
                "event_sensitive_keywords": item.event_sensitive_keywords,
                "event_match_keywords": item.event_match_keywords,
                "direct_match_aliases": item.direct_match_aliases,
                "relation_entities": item.relation_entities,
                "industry_nodes": item.industry_nodes,
                "benefit_direction_keywords": item.benefit_direction_keywords,
                "risk_direction_keywords": item.risk_direction_keywords,
                "summary_for_matching": item.summary_for_matching,
                "match_priority": item.match_priority,
                "confidence": item.confidence,
            }
        )
    return payload


def build_match_input(event: Task2Event, candidate_profiles: Iterable[CompanyProfile], max_match_results: int) -> str:
    event_payload = {
        "sample_id": event.sample_id,
        "title": event.title,
        "source_site": event.source_site,
        "url": event.url,
        "event_type": event.event_type,
        "event_summary": event.event_summary,
        "duration_type": event.duration_type,
        "event_attribute": event.event_attribute,
        "keyword": event.keyword,
        "heat": event.heat,
        "keyword_score": event.keyword_score,
        "scale_score": event.scale_score,
        "event_intensity": event.event_intensity,
        "influence_range": event.influence_range,
        "attribute_score": event.attribute_score,
        "impact_direction": event.impact_direction,
        "correlation_logic": event.correlation_logic,
        "classification_reason": event.classification_reason,
    }
    return TASK2_MATCH_PROMPT_TEMPLATE.format(
        system=TASK2_SYSTEM_PROMPT,
        max_match_results=max_match_results,
        event_json=json.dumps(event_payload, ensure_ascii=False, indent=2),
        candidate_profiles_json=json.dumps(
            _candidate_profile_payload(candidate_profiles),
            ensure_ascii=False,
            indent=2,
        ),
        sample_id=event.sample_id,
        title=event.title,
        event_type=event.event_type,
    )
