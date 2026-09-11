"""Two-tower encoders and bidirectional attention with gated fusion."""

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


class Time2Vec(nn.Module):
    """One linear time component and seven learned periodic components."""

    def __init__(self, kernel_size: int = TIME_DIM):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        self.k = kernel_size
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.W = nn.Parameter(torch.randn(kernel_size - 1))
        self.b = nn.Parameter(torch.randn(kernel_size - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        linear = self.w0 * t + self.b0
        periodic = torch.sin(t * self.W.view(1, 1, -1) + self.b.view(1, 1, -1))
        return torch.cat([linear, periodic], dim=-1)


class LSTMAttentionEncoder(nn.Module):
    """Encode one historical window with LSTM, Transformer, and attention pooling."""

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        dropout=0.3,
        time_dim: int = TIME_DIM,
        num_layers: int = 6,
    ):
        super().__init__()
        self.time2vec = Time2Vec(kernel_size=time_dim)

        self.total_lstm_layers = max(1, num_layers - 2)
        self.transformer_layers = min(2, max(1, num_layers - self.total_lstm_layers))

        self.lstm = nn.LSTM(
            input_dim + time_dim,
            hidden_dim,
            batch_first=True,
            num_layers=self.total_lstm_layers,
            dropout=dropout if self.total_lstm_layers > 1 else 0.0,
        )

        nhead = 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.transformer_layers
        )

        self.attn_fc = nn.Linear(hidden_dim, 1)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x_seq: torch.Tensor, t_seq: torch.Tensor) -> torch.Tensor:
        t_emb = self.time2vec(t_seq)
        x_in = torch.cat([x_seq, t_emb], dim=-1)
        lstm_out, _ = self.lstm(x_in)
        trans_out = self.transformer(lstm_out)
        attn_weights = torch.softmax(self.attn_fc(trans_out), dim=1)
        weighted = (trans_out * attn_weights).sum(dim=1)
        weighted = self.out_dropout(weighted)
        return weighted


class MultiScaleLSTMAttention(nn.Module):
    """Fuse per-window representations using learned softmax weights."""

    def __init__(
        self,
        input_dim=TAB_FEAT_DIM,
        hidden_dim=64,
        dropout=0.3,
        time_dim: int = TIME_DIM,
        num_layers: int = 6,
        window_sizes=(1, 5, 10),
    ):
        super().__init__()
        self.window_sizes = tuple(window_sizes)
        self.scales = [str(w) for w in self.window_sizes]
        self.encoders = nn.ModuleDict(
            {
                s: LSTMAttentionEncoder(
                    input_dim,
                    hidden_dim,
                    dropout=dropout,
                    time_dim=time_dim,
                    num_layers=num_layers,
                )
                for s in self.scales
            }
        )
        self.scale_weight = nn.Parameter(torch.ones(len(self.scales)))

    def fuse_feats(self, feats: List[torch.Tensor]) -> torch.Tensor:
        weights = torch.softmax(self.scale_weight, dim=0)
        stacked = torch.stack(feats, dim=0)
        fused = torch.sum(stacked * weights.view(-1, 1, 1), dim=0)
        return fused

    def forward(self, window_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]]):
        feats = []
        for w in self.window_sizes:
            x, t = window_dict[w]
            f = self.encoders[str(w)](x, t)
            feats.append(f)
        fused = self.fuse_feats(feats)
        return fused

    def encode_single(
        self, window_dict_single: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        feats = []
        device = next(self.parameters()).device
        for w in self.window_sizes:
            x, t = window_dict_single[w]
            x = x.unsqueeze(0).to(device)
            t = t.unsqueeze(0).to(device)
            f = self.encoders[str(w)](x, t)
            feats.append(f.squeeze(0))

        weights = torch.softmax(self.scale_weight, dim=0)
        stacked = torch.stack(feats, dim=0)
        weights_expanded = weights.unsqueeze(1)
        weighted_feats = stacked * weights_expanded
        fused = torch.sum(weighted_feats, dim=0)
        return fused


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        attn_out, _ = self.attention(query, key, value)
        out = self.norm(query + self.dropout(attn_out))
        return out


class CurrentTower(nn.Module):
    """Project concatenated semantic (768) and structural (1536) vectors."""

    def __init__(
        self,
        semantic_dim: int = SEMANTIC_DIM,
        struct_dim: int = STRUCT_DIM,
        hidden_dims: Tuple[int, ...] = (512, 256),
        output_dim: int = CURRENT_EMB_DIM,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.struct_dim = struct_dim
        self.input_dim = semantic_dim + struct_dim

        layers = []
        prev_dim = self.input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, semantic: torch.Tensor, struct: torch.Tensor) -> torch.Tensor:
        if struct.size(-1) != self.struct_dim:
            raise ValueError(f"Unexpected struct dimension: {struct.size(-1)}")

        x = torch.cat([semantic, struct], dim=-1)
        return self.encoder(x)


class GMU(nn.Module):
    """Learn elementwise retention gates for two representations."""

    def __init__(self, dim1: int, dim2: int, output_dim: int = None):
        super().__init__()
        self.output_dim = output_dim if output_dim is not None else max(dim1, dim2)

        self.W1 = nn.Linear(dim1, self.output_dim, bias=True)
        self.W2 = nn.Linear(dim2, self.output_dim, bias=True)
        self.Wz = nn.Linear(dim1 + dim2, self.output_dim, bias=True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)

        h1 = self.tanh(self.W1(x1))
        h2 = self.tanh(self.W2(x2))
        concat_input = torch.cat([x1, x2], dim=-1)
        z = self.sigmoid(self.Wz(concat_input))
        output = z * h1 + (1 - z) * h2
        return output

    def get_gate_values(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)
        concat_input = torch.cat([x1, x2], dim=-1)
        return self.sigmoid(self.Wz(concat_input))


class DualTowerWithGMU(nn.Module):
    """Refine both tower vectors through simultaneous bidirectional interaction."""

    def __init__(
        self,
        temporal_input_dim: int = TAB_FEAT_DIM,
        temporal_hidden_dim: int = TEMP_EMB_DIM,
        semantic_dim: int = SEMANTIC_DIM,
        struct_dim: int = STRUCT_DIM,
        current_hidden_dims: Tuple[int, ...] = (512, 256),
        current_output_dim: int = CURRENT_EMB_DIM,
        d_model: int = TEMP_EMB_DIM,
        nhead: int = 4,
        dropout: float = 0.3,
        num_blocks: int = 1,
        window_sizes=(1, 5, 10),
    ):
        super().__init__()

        assert (
            temporal_hidden_dim == current_output_dim == d_model
        ), f"Temporal({temporal_hidden_dim}) and Current({current_output_dim}) must match d_model({d_model})"

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

        self.num_blocks = num_blocks
        self.cross_attn_t2c_list = nn.ModuleList(
            [
                CrossAttention(d_model=d_model, nhead=nhead, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.cross_attn_c2t_list = nn.ModuleList(
            [
                CrossAttention(d_model=d_model, nhead=nhead, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.gmu_T_list = nn.ModuleList(
            [
                GMU(dim1=d_model, dim2=d_model, output_dim=d_model)
                for _ in range(num_blocks)
            ]
        )
        self.gmu_C_list = nn.ModuleList(
            [
                GMU(dim1=d_model, dim2=d_model, output_dim=d_model)
                for _ in range(num_blocks)
            ]
        )

        self.classifier = nn.Linear(d_model * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

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
        C = self.current_encoder(semantic, struct)
        C = C.squeeze(0)

        for i in range(self.num_blocks):
            T_seq = T.view(1, 1, -1)
            C_seq = C.view(1, 1, -1)

            T_cross = (
                self.cross_attn_t2c_list[i](T_seq, C_seq, C_seq).squeeze(0).squeeze(0)
            )
            C_cross = (
                self.cross_attn_c2t_list[i](C_seq, T_seq, T_seq).squeeze(0).squeeze(0)
            )

            T = self.gmu_T_list[i](T.unsqueeze(0), T_cross.unsqueeze(0)).squeeze(0)
            C = self.gmu_C_list[i](C.unsqueeze(0), C_cross.unsqueeze(0)).squeeze(0)

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
