import torch
import torch.nn as nn

class LSTMRacePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(LSTMRacePredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Multi-task heads
        self.fc_regression = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.fc_classification = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, features)
        # Using the mask to gather the last actual timestep per sequence, or just taking the output of the last timestep
        
        lstm_out, (hn, cn) = self.lstm(x)
        
        # We want the output at the last valid timestep for each batch element.
        # But for simplicity, we just take the last timestep of the sequence (seq_len=10)
        # Since we left-pad, the final step is always valid data for the sequence (even if sequence is < 10, the valid data is at the end).
        
        last_out = lstm_out[:, -1, :]
        
        expected_finish = self.fc_regression(last_out)
        top10_prob = self.fc_classification(last_out)
        
        return expected_finish, top10_prob
