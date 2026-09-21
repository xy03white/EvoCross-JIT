from typing import Dict, Tuple, List
import torch
from torch import nn
from .config import (
    TAB_FEAT_DIM,
    SEMANTIC_DIM,
    STRUCT_DIM,
    TEMP_EMB_DIM,
    CURRENT_EMB_DIM,
    TIME_DIM,
)

from .models import MultiScaleLSTMAttention, CurrentTower, GMU, CrossAttention


class AttentionOnlyBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.3):
        super().__init__()
        self.cross_t2c = CrossAttention(d_model, nhead, dropout)
        self.cross_c2t = CrossAttention(d_model, nhead, dropout)

    def forward(
        self, T: torch.Tensor, C: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T_seq = T.view(1, 1, -1)
        C_seq = C.view(1, 1, -1)
        T_new = self.cross_t2c(T_seq, C_seq, C_seq).squeeze(0).squeeze(0)
        C_new = self.cross_c2t(C_seq, T_seq, T_seq).squeeze(0).squeeze(0)
        return T_new, C_new


class GMUOnlyBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.3):
        super().__init__()
        self.gmu_T = GMU(dim1=d_model, dim2=d_model, output_dim=d_model)
        self.gmu_C = GMU(dim1=d_model, dim2=d_model, output_dim=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, T: torch.Tensor, C: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T_new = self.gmu_T(T.unsqueeze(0), C.unsqueeze(0)).squeeze(0)
        C_new = self.gmu_C(C.unsqueeze(0), T.unsqueeze(0)).squeeze(0)
        return T_new, C_new


class DualTowerModel(nn.Module):
    def __init__(
        self,
        window_sizes: List[int],
        interaction_type: str,
        num_blocks: int = 1,
        temporal_input_dim: int = TAB_FEAT_DIM,
        temporal_hidden_dim: int = TEMP_EMB_DIM,
        semantic_dim: int = SEMANTIC_DIM,
        struct_dim: int = STRUCT_DIM,
        current_hidden_dims: Tuple[int, ...] = (512, 256),
        current_output_dim: int = CURRENT_EMB_DIM,
        d_model: int = TEMP_EMB_DIM,
        nhead: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.temporal_encoder = MultiScaleLSTMAttention(
            input_dim=temporal_input_dim,
            hidden_dim=temporal_hidden_dim,
            dropout=dropout,
            time_dim=TIME_DIM,
            num_layers=6,
            window_sizes=window_sizes,
        )
        self.current_encoder = CurrentTower(
            semantic_dim=semantic_dim,
            struct_dim=struct_dim,
            hidden_dims=current_hidden_dims,
            output_dim=current_output_dim,
            dropout=dropout,
        )
        self.interaction_type = interaction_type
        self.num_blocks = num_blocks
        self.d_model = d_model

        if interaction_type == "concat":
            self.interaction_blocks = nn.ModuleList()
        elif interaction_type == "attention_only":
            self.interaction_blocks = nn.ModuleList(
                [AttentionOnlyBlock(d_model, nhead, dropout) for _ in range(num_blocks)]
            )
        elif interaction_type == "gmu_only":
            self.interaction_blocks = nn.ModuleList(
                [GMUOnlyBlock(d_model, dropout) for _ in range(num_blocks)]
            )
        else:
            raise ValueError(f"Unknown interaction_type: {interaction_type}")

        self.classifier = nn.Linear(d_model * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward_single(
        self,
        window_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        semantic: torch.Tensor,
        struct: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T = self.temporal_encoder.encode_single(window_dict)
        T = T.view(-1)
        if semantic.dim() == 1:
            semantic = semantic.unsqueeze(0)
        if struct.dim() == 1:
            struct = struct.unsqueeze(0)
        C = self.current_encoder(semantic, struct).squeeze(0)

        if self.interaction_type == "concat":
            pass
        else:
            for block in self.interaction_blocks:
                T, C = block(T, C)

        combined = torch.cat([T, C], dim=-1)
        combined = self.dropout(combined.unsqueeze(0))
        logits = self.classifier(combined).squeeze(0).squeeze(0)
        return T, C, logits

    def predict_proba(
        self,
        window_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        semantic: torch.Tensor,
        struct: torch.Tensor,
    ) -> float:
        self.eval()
        with torch.no_grad():
            _, _, logits = self.forward_single(window_dict, semantic, struct)
            prob = torch.sigmoid(logits).item()
        return prob


class TemporalOnlyModel(nn.Module):
    def __init__(
        self,
        window_sizes: List[int],
        temporal_input_dim: int = TAB_FEAT_DIM,
        temporal_hidden_dim: int = TEMP_EMB_DIM,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.temporal_encoder = MultiScaleLSTMAttention(
            input_dim=temporal_input_dim,
            hidden_dim=temporal_hidden_dim,
            dropout=dropout,
            time_dim=TIME_DIM,
            num_layers=6,
            window_sizes=window_sizes,
        )
        self.classifier = nn.Linear(temporal_hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward_single(
        self, window_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        T = self.temporal_encoder.encode_single(window_dict)
        T = T.view(-1)
        T = self.dropout(T.unsqueeze(0))
        logits = self.classifier(T).squeeze(0).squeeze(0)
        return logits

    def predict_proba(
        self, window_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    ) -> float:
        self.eval()
        with torch.no_grad():
            logits = self.forward_single(window_dict)
            prob = torch.sigmoid(logits).item()
        return prob


class CurrentOnlyModel(nn.Module):
    def __init__(
        self,
        semantic_dim: int = SEMANTIC_DIM,
        struct_dim: int = STRUCT_DIM,
        current_hidden_dims: Tuple[int, ...] = (512, 256),
        current_output_dim: int = CURRENT_EMB_DIM,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.current_encoder = CurrentTower(
            semantic_dim=semantic_dim,
            struct_dim=struct_dim,
            hidden_dims=current_hidden_dims,
            output_dim=current_output_dim,
            dropout=dropout,
        )
        self.classifier = nn.Linear(current_output_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward_single(
        self, semantic: torch.Tensor, struct: torch.Tensor
    ) -> torch.Tensor:
        if semantic.dim() == 1:
            semantic = semantic.unsqueeze(0)
        if struct.dim() == 1:
            struct = struct.unsqueeze(0)
        C = self.current_encoder(semantic, struct).squeeze(0)
        C = self.dropout(C.unsqueeze(0))
        logits = self.classifier(C).squeeze(0).squeeze(0)
        return logits

    def predict_proba(self, semantic: torch.Tensor, struct: torch.Tensor) -> float:
        self.eval()
        with torch.no_grad():
            logits = self.forward_single(semantic, struct)
            prob = torch.sigmoid(logits).item()
        return prob
