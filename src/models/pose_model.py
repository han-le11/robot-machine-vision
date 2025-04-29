import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionLSTM(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=128, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out  # shape: [batch, num_classes]

class SpatialAttention(nn.Module):
    def __init__(self, keypoint_dim=32, hidden_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(keypoint_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, 75, keypoint_dim]
        scores = self.attn(x)                         # [batch, seq_len, 75, 1]
        weights = torch.softmax(scores, dim=2)        # softmax over 75 joints
        attended = (weights * x).sum(dim=2)           # [batch, seq_len, keypoint_dim]
        return attended, weights

class TemporalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        scores = self.fc(x)                      # [batch_size, seq_len, 1]
        weights = torch.softmax(scores, dim=1)   # softmax over sequence
        context = (weights * x).sum(dim=1)        # weighted sum over time
        return context, weights

class ActionLSTMWithSpatialAttention(nn.Module):
    def __init__(self, num_classes, keypoint_dim=4, proj_dim=32, attn_hidden=64, lstm_hidden=128, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(keypoint_dim, proj_dim)             # 4 → 32
        self.joint_embed = nn.Parameter(torch.randn(75, proj_dim))  # Learnable joint identity
        self.spatial_attn = SpatialAttention(keypoint_dim=proj_dim, hidden_dim=attn_hidden)
        self.lstm = nn.LSTM(input_size=proj_dim, hidden_size=lstm_hidden, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, 300]
        x = x.view(x.size(0), x.size(1), 75, 4)            # → [batch, seq_len, 75, 4]
        x = self.proj(x)                                   # → [batch, seq_len, 75, 32]
        x = x + self.joint_embed                          # broadcast joint embedding
        attended, _ = self.spatial_attn(x)                # → [batch, seq_len, 32]
        attended = self.dropout(attended)
        lstm_out, _ = self.lstm(attended)                 # → [batch, seq_len, hidden]
        lstm_out = self.dropout(lstm_out)
        final_hidden = lstm_out[:, -1, :]                 # take last hidden state
        out = self.fc(final_hidden)                       # → [batch, num_classes]
        return out
    
class ActionLSTMWithSpatioTemporalAttention(nn.Module):
    def __init__(self, num_classes, keypoint_dim=4, proj_dim=32, attn_hidden=64, lstm_hidden=128, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(keypoint_dim, proj_dim)
        self.joint_embed = nn.Parameter(torch.randn(75, proj_dim))
        self.spatial_attn = SpatialAttention(keypoint_dim=proj_dim, hidden_dim=attn_hidden)
        self.lstm = nn.LSTM(input_size=proj_dim, hidden_size=lstm_hidden, batch_first=True)
        self.temporal_attn = TemporalAttention(input_dim=lstm_hidden, hidden_dim=attn_hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 75, 4)
        x = self.proj(x)
        x = x + self.joint_embed
        spatial_attended, _ = self.spatial_attn(x)
        spatial_attended = self.dropout(spatial_attended)
        lstm_out, _ = self.lstm(spatial_attended)
        lstm_out = self.dropout(lstm_out)
        temporal_context, _ = self.temporal_attn(lstm_out)
        out = self.fc(temporal_context)
        return out