# Math_match — 金融事件驱动股票策略系统

基于财经新闻事件的A股量化选股系统，覆盖从数据采集、事件检测/分类、公司关联挖掘、涨跌预测到选股执行的全流程。

## 项目架构

```
Web_scrope/        → 财经新闻爬虫（东方财富、中国政府网、巨潮资讯网、第一财经）
task1/             → 事件检测 + 事件分类 + 结构化抽取 + 伪标签迭代
task2/             → 公司画像构建 + 事件-公司语义匹配 + LLM关联判定
task3/             → LSTM模型（PaperEventLSTM）训练 + 推理
task4/             → 多事件融合选股 + 资金分配 + 策略回测
run_*.py           → 各Pipeline入口脚本
plot_*.py          → 可视化脚本
outputs/           → 核心输出数据
```

## 快速开始

```bash
pip install -r requirements.txt
```

### Task1: 事件检测与分类

```bash
python run_live_classification.py --input <news_json> --output outputs/
```

### Task2: 事件-公司关联

```bash
python run_live_task2.py --events <classification_results> --output outputs/
python run_live_task2_semantic.py --events <classification_results> --output outputs/
```

### Task3+4: 预测与选股

```bash
python run_live_task34_real.py --events <events> --companies <profiles> --output outputs/
```

## 核心数据说明

| 文件 | 说明 |
|:---|:---|
| outputs/live_classification_eastmoney30/classification_results.json | 30条新闻事件分类结果 |
| outputs/live_task2_match/event_company_links.json | 事件-公司LLM关联结果 |
| outputs/live_task2_match/normalized_hs300.json | 沪深300成分股列表 |
| outputs/live_task2_semantic/s0_topk.json | 语义匹配Top-K结果 |
| outputs/live_task4/real_predictions.json | LSTM预测结果（pred_prob + pred_return） |
| outputs/live_task4/real_selected_stocks.json | 最终选股结果 |
| outputs/top45_event_company_4_11_to_4_14.json | Top45事件关联公司 |

## 模型

- **事件分类**：LLM + 规则混合（事件属性评分公式）
- **语义匹配**：BGE-large-zh-v1.5 余弦相似度
- **涨跌预测**：PaperEventLSTM（Char-level TF-IDF + 数值特征 → 双头输出）
- **选股融合**：概率叠加截断 + 收益线性累加 → Score排序
