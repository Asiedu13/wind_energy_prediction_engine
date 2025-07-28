import torch.nn as nn

class PowerRNN(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=2, dropout=0.2):
        super(PowerRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()
