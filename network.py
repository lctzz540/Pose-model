import torch
import torch.nn as nn


class LSTMPoseClassifier(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.2
    ):
        super(LSTMPoseClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.bn = nn.BatchNorm1d(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, teacher_forcing=False):
        batch_size, sequence_length, input_size = x.size()
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(
            device=x.device
        )
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(
            device=x.device
        )

        if teacher_forcing:
            out, _ = self.lstm(x, (h0, c0))
            out = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            out = []
            h, c = h0, c0
            for t in range(sequence_length):
                xt = x[:, t, :].unsqueeze(1)
                ht, ct = self.lstm(xt, (h, c))
                ht = self.bn(ht.permute(0, 2, 1)).permute(0, 2, 1)
                ht = self.dropout(ht)
                h, c = ht, ct
                out.append(ht)
            out = torch.cat(out, dim=1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out[:, -1, :])
        return out
