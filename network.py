import torch.nn as nn


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMPredictor, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Apply LSTM layers
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)

        # Flatten the sequence dimension
        out = out.reshape(out.shape[0], -1)

        # Apply the fully-connected layer
        out = self.fc(out)

        return out
