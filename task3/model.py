import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class PaperEventLSTM(nn.Module):
    """
    Paper-aligned three-branch multi-task model.

    - Event branch: event text + structured event features
    - Company branch: company profile text + structured company features
    - Time branch: two-day stock sequence encoded by LSTM + delta features
    - Heads: classification logit + regression output
    """

    def __init__(
        self,
        event_text_dim: int,
        event_num_dim: int,
        company_text_dim: int,
        company_num_dim: int,
        time_input_dim: int = 10,
        delta_dim: int = 10,
        text_hidden_dim: int = 64,
        num_hidden_dim: int = 32,
        time_hidden_dim: int = 32,
        fusion_hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.event_text_encoder = MLPBlock(event_text_dim, text_hidden_dim, text_hidden_dim, dropout)
        self.event_num_encoder = MLPBlock(event_num_dim, num_hidden_dim, num_hidden_dim, dropout)

        self.company_text_encoder = MLPBlock(company_text_dim, text_hidden_dim, text_hidden_dim, dropout)
        self.company_num_encoder = MLPBlock(company_num_dim, num_hidden_dim, num_hidden_dim, dropout)

        self.time_lstm = nn.LSTM(
            input_size=time_input_dim,
            hidden_size=time_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.delta_encoder = MLPBlock(delta_dim, num_hidden_dim, num_hidden_dim, dropout)
        self.time_fusion = MLPBlock(time_hidden_dim + num_hidden_dim, 48, 48, dropout)

        event_dim = text_hidden_dim + num_hidden_dim
        company_dim = text_hidden_dim + num_hidden_dim
        time_dim = 48
        fusion_input_dim = event_dim + company_dim + time_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(fusion_hidden_dim, 1)
        self.regressor = nn.Linear(fusion_hidden_dim, 1)

    def forward(
        self,
        event_text,
        event_num,
        company_text,
        company_num,
        time_seq,
        delta_feat,
    ):
        event_text_repr = self.event_text_encoder(event_text)
        event_num_repr = self.event_num_encoder(event_num)
        event_repr = torch.cat([event_text_repr, event_num_repr], dim=-1)

        company_text_repr = self.company_text_encoder(company_text)
        company_num_repr = self.company_num_encoder(company_num)
        company_repr = torch.cat([company_text_repr, company_num_repr], dim=-1)

        _, (time_hidden, _) = self.time_lstm(time_seq)
        time_hidden = time_hidden[-1]
        delta_repr = self.delta_encoder(delta_feat)
        time_repr = self.time_fusion(torch.cat([time_hidden, delta_repr], dim=-1))

        fused = self.fusion(torch.cat([event_repr, company_repr, time_repr], dim=-1))
        cls_logit = self.classifier(fused).squeeze(-1)
        reg_value = self.regressor(fused).squeeze(-1)

        return {
            "cls_logit": cls_logit,
            "reg_value": reg_value,
            "fused": fused,
        }


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
