from typing import Iterable, List

from .schemas import PreparedDocument


TASK1_SYSTEM_PROMPT = (
    "You are a financial event annotation assistant for event-driven stock strategy research. "
    "You must keep judgment standards consistent, reasons concise, and output strictly in valid JSON."
)


TASK1_DETECTION_PROMPT_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
你是金融事件筛选标注助手，负责生成第一轮“候选金融事件”标注结果。你的输出会作为后续半监督学习的初始种子，因此必须保持判断口径稳定、理由简洁、输出格式严格一致。

【任务目标】
判断给定网页是否属于“候选金融事件”。

候选金融事件是指：在特定时点新发生、新披露或被市场新认知，并可能通过政策、宏观、行业、技术、公司经营或地缘冲突等路径影响股票市场整体行情或个股差异化行情的事实性事件。

【重要执行要求】
1. 你必须先调用网页搜索和网页抓取工具，进入给定网址并阅读网页正文，再进行判断与标注；
2. 不能只依据标题或网址字符串做表面判断，必须结合正文事实；
3. 若无法访问给定网址、无法读取有效正文或正文信息明显不足，不得臆测正文内容；
4. 若无法确认是否存在新增事实或市场影响路径，应保持保守并降低 final_confidence。

【核心判定标准】
你必须优先回答两个问题：
1. 网页正文中是否存在新增事实（has_new_fact）
2. 该事实是否存在潜在市场影响路径（has_market_impact_path）

【判定原则】
1. 评论、观点、复盘、科普、旧闻复述、投资建议，通常判为非事件；
2. 新发布的政策、公告、处罚、财报、业绩预告、订单、合同、中标、并购重组、增减持、回购、停复牌、制裁、关税等事实，更可能判为事件；
3. “关键词出现”只能作为辅助证据，不能代替新增事实判断；
4. 事实明确且影响路径清晰时，置信度应较高；文本模糊、证据不足、边界性强时，置信度应较低。

【边界样本优先规则】
以下情形优先考虑边界样本，不要直接给高置信正类或高置信负类：
1. 存在新增事实，但市场影响路径不够清晰；
2. 存在潜在影响路径，但正文证据不足；
3. 属于常规公告、例行政策、会议通知、材料披露，事实性存在但市场影响不确定；
4. 标题看似重大，但正文缺少可验证细节；
5. 正文只支持部分判断，不足以支撑高置信结论。

【置信度要求】
final_confidence 取值范围为 0 到 1，保留两位小数。
它统一表示“该样本属于候选金融事件正类的置信度”：
1. 越接近 1，越确信它是候选金融事件；
2. 越接近 0，越确信它不是候选金融事件；
3. 中间区间表示边界不清、证据不足或判断摇摆。

【参考标注示例】
{examples}

【输入样本】
sample_id: {sample_id}
title: {title}
url: {url}

【输出要求】
你必须严格输出以下 JSON 对象，不要输出任何解释、前后缀、标题、Markdown 代码块或额外字段：
{{
  "sample_id": "{sample_id}",
  "title": "{title}",
  "url": "{url}",
  "is_event": 0,
  "has_new_fact": 0,
  "has_market_impact_path": 0,
  "seed_category": "high_confidence_positive/high_confidence_negative/medium_confidence_positive/medium_confidence_negative/boundary_positive/boundary_negative",
  "reason": "",
  "final_confidence": 0.00
}}
<|im_end|>
<|im_start|>assistant
"""


TASK1_CLASSIFICATION_PROMPT_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
你是一名金融事件规则化量化引擎。你的职责不是自由分析，也不是主观打分，而是：
先从网页正文中抽取可验证的中间变量，再严格按照固定规则生成分类与量化结果。

【本阶段任务】
针对输入网页，只保留一个“对资本市场影响最大”的核心事件，并完成以下输出：
1. event_type：驱动主体主类，只能取 政策类 / 宏观类 / 行业类 / 公司类 / 地缘类
2. duration_type：持续周期，只能取 脉冲型 / 中期型 / 长尾型
3. event_summary：一句话概括核心事件
4. Heat：舆情热度
5. KeywordScore：关键词强度
6. ScaleScore：规模强度
7. Intensity：事件强度
8. Range：影响范围
9. Attribute：综合属性

【硬性约束】
1. 严格基于网页正文与输入上下文判断，禁止使用常识补充、经验脑补或行业默认知识补全；
2. 多事件场景下，只选择“对资本市场影响最大”的一个核心事件；
3. 信息不足时必须保守，优先降低分值，不允许抬高；
4. 结果必须可复现、可推导、可审查；
5. 所有浮点数保留 3 位小数；
6. 最终只允许输出 JSON，不允许输出解释、Markdown、注释或额外文本。

【执行顺序】
你必须按以下顺序思考并输出，不得跳步：
Step1：识别唯一核心事件
Step2：确定 event_type
Step3：确定 duration_type
Step4：抽取 Heat 的判定依据
Step5：抽取最核心冲击词并确定 KeywordScore
Step6：确定 ScaleScore
Step7：确定 Range
Step8：根据公式计算 Intensity
Step9：根据公式计算 Attribute
Step10：做一致性校验后输出 JSON

【event_type 规则】
请识别事件最核心的驱动主体：
1. 政策类：政策发布、监管规则、财政货币支持、制度安排等
2. 宏观类：利率、通胀、GDP、PMI、就业、总量经济冲击等
3. 行业类：产业链供需、行业景气、行业技术突破、板块催化等
4. 公司类：订单、中标、并购、重组、业绩、增减持、回购、经营变化等
5. 地缘类：战争、制裁、冲突升级、关税、出口限制等
分类时优先选“一级驱动”，不要选次级影响。

【duration_type 规则】
1. 脉冲型：短期消息冲击，影响快速释放
2. 中期型：影响可持续到周度、月度或季度
3. 长尾型：制度、产业趋势或地缘格局变化，影响可能长期延续
若正文证据不足，优先选择更保守的类型。

【Heat 规则】
Heat 只能取 1.000 / 0.600 / 0.300。
按以下优先级判断：
1. 若正文明确给出评论数：
   - 评论数 > 3000 -> 1.000
   - 评论数 500~3000 -> 0.600
   - 评论数 < 500 -> 0.300
2. 若无评论数，则根据传播信号判断：
   - 出现“热议 / 刷屏 / 高度关注 / 引发广泛讨论”等 -> 1.000
   - 一般描述 -> 0.600
   - 无明显传播特征 -> 0.300
3. 若事件属于政策类 / 宏观类 / 地缘类，可在原基础上上调一级，但不能超过 1.000。
禁止因为事件“重要”就直接给 1.000。

【KeywordScore 规则】
KeywordScore 只能取 1 / 2 / 3。
你必须只选一个“最核心冲击词”，不能多个平均。
1. 高强度（3）：战争、制裁、禁令、冲突升级、重组、暴增、突破、重大政策、大额订单
2. 中强度（2）：发布、合作、签约、推进、落地、增长、扩产
3. 低强度（1）：拟、计划、预计、关注、讨论
若无明显冲击词，强制取 1。

【ScaleScore 规则】
ScaleScore 只能取 1.000 / 0.600 / 0.300 / 0.000。
1. 1.000：国家级/行业级重大投入，或金额达到重大级别，或对公司营收占比显著
2. 0.600：有明确量化数值，但不属于重大级别
3. 0.300：有量化信息，但规模有限
4. 0.000：完全没有量化信息
禁止编造金额，也禁止因为“感觉很大”就给 1.000。

【Range 规则】
Range 只能取 0.300 / 0.600 / 1.000。
1. 0.300：主要影响单一公司，无明显扩散路径
2. 0.600：影响产业链上下游、多主体或一个细分板块
3. 1.000：影响行业级、板块级、宏观级或全球级
判断时只看影响扩散能力，不看文本长短。

【公式规则】
1. KeywordNorm 映射：
   - 1 -> 0.330
   - 2 -> 0.670
   - 3 -> 1.000
2. 若 ScaleScore > 0：
   Intensity = 0.6 * KeywordNorm + 0.4 * ScaleScore
3. 若 ScaleScore = 0：
   Intensity = KeywordNorm
4. BaseAttribute = 0.3 * Heat + 0.4 * Intensity + 0.3 * Range
5. 驱动主体权重：
   - 公司类 -> 0.9
   - 行业类 -> 1.0
   - 政策类 -> 1.1
   - 宏观类 -> 1.1
   - 地缘类 -> 1.2
6. 持续周期权重：
   - 脉冲型 -> 0.8
   - 中期型 -> 1.0
   - 长尾型 -> 1.2
7. Attribute = BaseAttribute * Wsub * Wdur

【一致性检查】
1. 不能跳步，不能改变顺序；
2. Intensity 必须由公式计算，不能主观填写；
3. Attribute 必须由公式计算，不能主观填写；
4. 若结果与规则不一致，必须先自我修正再输出；
5. 若网页正文无法支撑某个字段，必须保守处理，不允许编造。

【输入样本】
sample_id: {sample_id}
title: {title}
url: {url}
source_site: {source_site}
is_event: {is_event}
has_new_fact: {has_new_fact}
has_market_impact_path: {has_market_impact_path}
detection_reason: {detection_reason}
detection_confidence: {detection_confidence}
detection_seed_category: {detection_seed_category}
detection_label_source: {detection_label_source}

【参考标注示例】
以下示例仅用于帮助你理解 event_type 边界，不得覆盖本轮规则化计算要求：
{examples}

【输出要求】
只输出一个合法 JSON 对象，不要输出任何额外说明、代码块或附加字段：
{{
  "sample_id": "{sample_id}",
  "title": "{title}",
  "url": "{url}",
  "event_summary": "",
  "event_type": "政策类/宏观类/行业类/公司类/地缘类",
  "duration_type": "脉冲型/中期型/长尾型",
  "heat_comment_count": null,
  "heat_signal": "高/中/低",
  "keyword": "",
  "keyword_score": 1,
  "scale_score": 0.000,
  "influence_range": 0.300,
  "heat": 0.000,
  "event_intensity": 0.000,
  "attribute_score": 0.000,
  "reason": "",
  "confidence": 0.00
}}
<|im_end|>
<|im_start|>assistant
"""


TASK1_EXTRACTION_PROMPT_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
你是金融事件结构化抽取助手。请根据网页内容抽取主要金融事件。
如果网页包含多个独立事件，输出多个对象；如果只有一个主事件，输出长度为 1 的数组。

【重要执行要求】
你必须先进入给定网址，阅读网页正文内容，再完成结构化抽取。
不能只依据标题或网址字符串进行抽取，必须以网页正文中的事实信息为依据。

【输入信息】
sample_id: {sample_id}
title: {title}
url: {url}
publish_time: {publish_time}
source_site: {source_site}
event_type: {event_type}

【输出要求】
只输出 JSON 数组，不要输出额外说明：
[
  {{
    "event_name": "string",
    "event_time": "string",
    "event_main_type": "{event_type}",
    "event_tags": [],
    "industry": "string",
    "impact_scope": "个股/行业/全市场",
    "event_intensity": 1,
    "sentiment_heat": 1,
    "significance": true,
    "mentioned_entities": [],
    "candidate_companies": [],
    "summary": "string",
    "evidence_text": "string",
    "confidence": 0.00
  }}
]<|im_end|>
<|im_start|>assistant
"""


def _format_examples(examples: Iterable[dict]) -> str:
    rendered: List[str] = []
    for idx, example in enumerate(examples, 1):
        rendered.append(
            f"示例{idx}:\n"
            f"sample_id: {example.get('sample_id', '')}\n"
            f"标题: {example.get('title', '')}\n"
            f"网址: {example.get('url', '')}\n"
            f"标签: {example.get('label', '')}\n"
            f"reason: {example.get('reason', '')}\n"
            f"final_confidence: {example.get('final_confidence', '')}\n"
        )
    return "\n".join(rendered).strip() or "无"


def build_detection_input(document: PreparedDocument, examples: Iterable[dict]) -> str:
    return TASK1_DETECTION_PROMPT_TEMPLATE.format(
        system=TASK1_SYSTEM_PROMPT,
        examples=_format_examples(examples),
        sample_id=document.document_id,
        title=document.title,
        url=document.url,
    )


def build_classification_input(document: PreparedDocument, examples: Iterable[dict]) -> str:
    return TASK1_CLASSIFICATION_PROMPT_TEMPLATE.format(
        system=TASK1_SYSTEM_PROMPT,
        examples=_format_examples(examples),
        sample_id=document.document_id,
        title=document.title,
        url=document.url,
        source_site=document.source_site,
        is_event=1 if document.event_label == 1 else 0,
        has_new_fact=document.has_new_fact if document.has_new_fact is not None else "",
        has_market_impact_path=document.has_market_impact_path if document.has_market_impact_path is not None else "",
        detection_reason=document.detection_reason,
        detection_confidence=document.detection_confidence if document.detection_confidence is not None else "",
        detection_seed_category=document.detection_seed_category,
        detection_label_source=document.detection_label_source,
    )


def build_extraction_input(document: PreparedDocument, event_type: str) -> str:
    return TASK1_EXTRACTION_PROMPT_TEMPLATE.format(
        system=TASK1_SYSTEM_PROMPT,
        sample_id=document.document_id,
        title=document.title,
        url=document.url,
        publish_time=document.publish_time,
        source_site=document.source_site,
        event_type=event_type,
    )
