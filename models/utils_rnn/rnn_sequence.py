import torch
import torch.nn as nn


# nn.Module that unfolds a RNN cell into a sequence
class RNNSequence(nn.Module):
    def __init__(self, rnn_cell):
        super().__init__()
        self.rnn_cell = rnn_cell

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        # init hidden state
        hidden_state = torch.zeros(
            (batch_size, self.rnn_cell.state_size), device=device)
        outputs = []
        for t in range(seq_len):
            inputs = x[:, t]
            new_output, hidden_state = self.rnn_cell.forward(inputs, hidden_state)
            outputs.append(new_output)
        outputs = torch.stack(outputs, dim=1)  # return entire sequence
        return outputs
