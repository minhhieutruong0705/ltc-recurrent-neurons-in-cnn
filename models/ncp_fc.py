from pyexpat import model
import torch
import torch.nn as nn
import torchinfo
import kerasncp as kncp
from kerasncp.torch import LTCCell


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


"""
NCP_FC utilizes the dynamicity of NCP to optimize the weight 
of conventional fully-connected layers of CNN classifiers.
Default values of NCP architecture are borrowed from the original work of NCP
(Neural circuit policies enabling auditable autonomy by Mathias Lechner et. al., Oct 2020)
"""


class NCP_FC(nn.Module):
    def __init__(self,
                 seq_len,
                 classes=2,
                 bi_directional=False,
                 sensory_neurons=32,
                 inter_neurons=12,
                 command_neurons=6,
                 motor_neurons=1,
                 sensory_outs=6,
                 inter_outs=4,
                 recurrent_dense=6,
                 motor_ins=6
                 ):
        super().__init__()

        # init
        self.seq_len = seq_len
        self.classes = classes
        self.bi_directional = bi_directional
        self.sensory_neurons = sensory_neurons
        self.inter_neurons = inter_neurons
        self.command_neurons = command_neurons
        self.motor_neurons = motor_neurons
        self.sensory_outs = sensory_outs
        self.inter_outs = inter_outs
        self.recurrent_dense = recurrent_dense
        self.motor_ins = motor_ins

        # ncp wiring
        wiring = kncp.wirings.NCP(
            inter_neurons=inter_neurons,
            command_neurons=command_neurons,
            motor_neurons=motor_neurons,
            sensory_fanout=sensory_outs,
            inter_fanout=inter_outs,
            recurrent_command_synapses=recurrent_dense,
            motor_fanin=motor_ins,
        )

        # forward
        ltc_cell_fwd = LTCCell(wiring=wiring, in_features=sensory_neurons)
        self.ltc_fwd_seq = RNNSequence(ltc_cell_fwd)
        # backward
        if self.bi_directional:
            ltc_cell_bwd = LTCCell(wiring=wiring, in_features=sensory_neurons)
            self.ltc_bwd_seq = RNNSequence(ltc_cell_bwd)

        # reduce ncp sequential outputs to number of classes
        directions = 2 if self.bi_directional else 1
        self.fc = nn.Linear(motor_neurons * seq_len * directions, classes)

    def extra_repr(self):
        return f"sensory_neurons={self.sensory_neurons}, " \
               f"inter_neurons={self.inter_neurons}, " \
               f"command_neurons={self.command_neurons}, " \
               f"motor_neurons={self.motor_neurons},\n" \
               f"sensory_outs={self.sensory_outs}, " \
               f"inter_outs={self.inter_outs}, " \
               f"recurrent_dense={self.recurrent_dense}, " \
               f"motor_ins={self.motor_ins},\n" \
               f"seq_len={self.seq_len}, " \
               f"classes={self.classes}, " \
               f"bi_directional={self.bi_directional}"

    def forward(self, x):
        # x: (B, S, C)
        x_fw = self.ltc_fwd_seq(x)
        if self.bi_directional:
            x = x.flip(dims=[1])  # backward input sequence
            x_bw = self.ltc_bwd_seq(x)
            x_bw = x_bw.flip(dims=[1])  # backward prediction for concatenation
        x = x_fw if not self.bi_directional else torch.cat((x_fw, x_bw), dim=-1)  # bi-directional concatenate
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


if __name__ == '__main__':
    x = torch.randn(8, 32, 64)  # (batch_size, sequence_length, features)
    model = NCP_FC(seq_len=32, classes=2, bi_directional=True, sensory_neurons=64)
    y = model(x)
    assert y.size() == (8, 2)
    print("[ASSERTION] NCP_FC OK!")
    print(model)
    torchinfo.summary(model=model, input_data=x, device="cpu")
