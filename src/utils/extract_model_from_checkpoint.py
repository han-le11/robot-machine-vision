import torch
import torch.nn as nn

# Config
CHECKPOINT_PATH = "checkpoint_epoch03_valacc_97.50.pt"
OUTPUT_PATH = "lstm_model.pt"

class ActionLSTM(nn.Module):
    def __init__(self, feature_dim=1280, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return torch.sigmoid(self.fc(h_n[-1])).squeeze(1)

model = ActionLSTM()
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
torch.save(model.state_dict(), OUTPUT_PATH)

print(f"Saved model weights to: {OUTPUT_PATH}")
