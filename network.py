import torch
import torch.nn as nn


class LSTMPoseClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMPoseClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, sequence_length, input_size = x.size()
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
            device=x.device
        )
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
            device=x.device
        )

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out
