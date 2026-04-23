"""
Task 3: LSTM-based Event-Driven Stock Prediction
Schemas — Task3Config, Sample, EventRecord
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Task3Config:
    # Paths
    events_path: str = (
        r"D:\Math_match\codes\outputs\classification_run_eastmoney400_v2\classification_results_new.json"
    )
    company_profiles_path: str = (
        r"D:\Math_match\codes\outputs\task2_profile_run\company_profiles.json"
    )
    assoc_matrix_path: str = (
        r"D:\Math_match\codes\outputs\task2_attribute_correction\assoc_matrix.csv"
    )
    stock_price_dir: str = r"D:\Math_match\codes\data\stock_prices"

    # Output
    output_dir: str = r"D:\Math_match\codes\outputs\task3"

    # Sample generation params
    top_k_assoc: int = 30          # 每事件保留 Top-K 关联公司
    min_assoc_threshold: float = 0.0  # Assoc_ij 最低阈值

    # CAR window
    car_window: int = 4             # CAR(4): 事件后4个交易日
    pre_event_days: int = 4        # 事件前窗口（用于特征提取）

    # Feature dimensions
    event_feat_dim: int = 12       # 事件特征维度
    profile_feat_dim: int = 20      # 公司画像维度
    price_feat_dim: int = 8        # 股价特征维度（每日前后拼接）

    # LSTM
    lstm_hidden: int = 64
    lstm_layers: int = 2
    dropout: float = 0.3

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class EventRecord:
    """单条事件的特征向量"""
    sample_id: str
    publish_time: str              # "4.11" ~ "4.14"（假设2026年）
    year: int = 2026
    event_type: str
    duration_type: str
    # 四项量化特征
    heat: float
    event_intensity: float
    influence_range: float
    attribute_score: float
    # 文本特征（可选，暂用统计特征）
    event_summary: str

    def to_feature_vector(self) -> list:
        """12维事件特征向量"""
        # 标准化到 [0, 1]（留待数据到位后用训练集统计）
        return [
            self.heat,
            self.event_intensity,
            self.influence_range,
            self.attribute_score,
            1.0 if self.event_type == "政策类" else 0.0,
            1.0 if self.event_type == "宏观类" else 0.0,
            1.0 if self.event_type == "行业类" else 0.0,
            1.0 if self.event_type == "公司类" else 0.0,
            1.0 if self.event_type == "地缘类" else 0.0,
            1.0 if self.duration_type == "脉冲型" else 0.0,
            1.0 if self.duration_type == "长尾型" else 0.0,
            1.0 if self.duration_type == "持续型" else 0.0,
        ]


@dataclass
class Sample:
    """单条样本：事件 × 公司"""
    sample_id: str
    stock_code: str
    stock_name: str
    assoc_score: float             # Assoc_ij
    event_feature: list           # 12维
    profile_feature: list          # 20维
    # 以下字段 stock 数据到位后填充
    price_feature: list = field(default_factory=list)   # 4日 × 2 = 8维
    car_4: float = None           # CAR(4)，待计算
    direction_label: int = None    # 1 if CAR>0 else 0

    def has_label(self) -> bool:
        return self.car_4 is not None
