# LSTM 训练数据说明

## 已生成文件
- `lstm_event_stock_dataset_v1.json`：可直接用于训练的总样本集
- `train_samples_v1.json`：训练集（事件日 20260411-20260412）
- `val_samples_v1.json`：验证集（事件日 20260413）
- `pending_inference_samples_v1.json`：待预测样本（事件日 20260414，因未来4交易日标签不完整未纳入训练）
- `dropped_samples_report_v1.json`：被剔除样本记录

## 本次采用的最标准可落地格式
基于你现有三个文件，采用“单事件-单公司”样本级格式：

- 事件信息
- 关联强度 `relation_score`
- 双交易日输入 `input_window_pre2`
- 差分特征 `delta_features`
- 未来4交易日窗口 `future_window_post4`
- 标签 `targets`

## 单条样本结构
```json
{
  "sample_id": "20260411_DOC-0310_601229",
  "event_date": "20260411",
  "event_trade_date": "20260410",
  "label_start_date": "20260413",
  "stock_code": "601229",
  "event_id": "DOC-0310",
  "event_title": "...",
  "event_summary": "...",
  "relation_score": 0.978934,
  "event_features": {
    "event_text": "...",
    "actual_event_date": "4.11",
    "date_source_field": "event_summary",
    "date_match": "2026年4月11日",
    "rank_within_day_top45": 1
  },
  "input_window_pre2": [
    {
      "date": "20260409",
      "open": 9.69,
      "close": 9.62,
      "high": 9.78,
      "low": 9.62,
      "volume": 265923.43,
      "amount": 257210.47,
      "amplitude": 1.64,
      "pct_change": -1.13,
      "change_amount": -0.11,
      "turnover_rate": null
    },
    {
      "date": "20260410",
      "open": 9.63,
      "close": 9.59,
      "high": 9.66,
      "low": 9.56,
      "volume": 356557.23,
      "amount": 342233.49,
      "amplitude": 1.04,
      "pct_change": -0.31,
      "change_amount": -0.03,
      "turnover_rate": null
    }
  ],
  "delta_features": {
    "open": -0.06,
    "close": -0.03,
    "high": -0.12,
    "low": -0.06,
    "volume": 90633.8,
    "amount": 85023.02,
    "amplitude": -0.6,
    "pct_change": 0.82,
    "change_amount": 0.08,
    "turnover_rate": null
  },
  "future_window_post4": [
    {"date": "20260413", "open": 9.59, "close": 9.58, "high": 9.61, "low": 9.52, "volume": 264314.18, "amount": 252525.45, "amplitude": 0.94, "pct_change": -0.1, "change_amount": -0.01, "turnover_rate": null},
    {"date": "20260414", "open": 9.58, "close": 9.79, "high": 9.82, "low": 9.57, "volume": 550309.23, "amount": 537023.93, "amplitude": 2.61, "pct_change": 2.19, "change_amount": 0.21, "turnover_rate": null},
    {"date": "20260415", "open": 9.81, "close": 9.88, "high": 9.9, "low": 9.76, "volume": 347137.78, "amount": 341435.23, "amplitude": 1.43, "pct_change": 0.92, "change_amount": 0.09, "turnover_rate": null},
    {"date": "20260416", "open": 9.88, "close": 9.8, "high": 9.97, "low": 9.79, "volume": 366977.05, "amount": 362110.09, "amplitude": 1.82, "pct_change": -0.81, "change_amount": -0.08, "turnover_rate": null}
  ],
  "targets": {
    "future_4day_return": 0.022965,
    "future_4day_up": 1
  },
  "status": "labeled"
}
```

## 标签定义
由于你没有市场基准指数，本次不计算 CAR(4)，改用可直接训练的替代标签：

- `future_4day_return = (future_window_post4[-1].close - future_window_post4[0].close) / future_window_post4[0].close`
- `future_4day_up = 1 if future_4day_return > 0 else 0`

## 时间对齐
- 输入窗口：事件触发前最近两个可用交易日
- 标签窗口：严格晚于事件自然日后的4个交易日

例如：
- 4.11 事件 -> 输入 4.09, 4.10；标签 4.13-4.16
- 4.12 事件 -> 输入 4.09, 4.10；标签 4.13-4.16
- 4.13 事件 -> 输入 4.10, 4.13；标签 4.14-4.17
- 4.14 事件 -> 因缺少后续完整4交易日标签，被放入 pending 文件

## 数据量
- 总可训练样本：130
- 训练集：89
- 验证集：41
- 待预测样本：42
- 剔除样本：8
