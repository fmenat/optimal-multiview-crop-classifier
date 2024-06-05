import torch
from torch import nn

from .base_encoders import Base_Encoder
from .temporal_pool import TemporalAttentionPool

#from https://github.com/jameschapman19/cca_zoo
class MLP(Base_Encoder):
    def __init__(
        self,
        feature_size: int,
        layer_sizes: tuple = None,
        activation=nn.ReLU, #LeakyReLU, GELU or nn.Tanh()
        dropout=0,
        batchnorm: bool=False,
        **kwargs,
    ):
        super(MLP, self).__init__()
        if layer_sizes is None:
            layer_sizes = (128,)
        layer_sizes = (feature_size,) + layer_sizes
        self.encoder_output = layer_sizes[-1]
        layers = []
        # other layers
        for l_id in range(len(layer_sizes) - 1):
            layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    activation(),
                    nn.BatchNorm1d(layer_sizes[l_id+1], affine=True) if batchnorm else nn.Identity(),
                    nn.Dropout(p=dropout) if dropout!=0 else nn.Identity(),
                )
            )
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return {"rep": self.layers(x)}

    def get_output_size(self):
        return self.encoder_output

class RNNet(Base_Encoder):
    def __init__(
        self,
        feature_size: int,
        layer_size: int = 128,
        dropout: float =0,
        num_layers: int = 1,
        bidirectional: bool = False,
        unit_type: str="gru",
        batchnorm: bool = False,
        temporal_pool: bool = False,
        **kwargs,
    ):
        super(RNNet, self).__init__()
        self.unit_type = unit_type.lower()
        self.feature_size = feature_size
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batchnorm = batchnorm
        self.temporal_pool = temporal_pool

        if self.unit_type == "gru":
            rnn_type_class = torch.nn.GRU
        elif self.unit_type == "lstm":
            rnn_type_class = torch.nn.LSTM
        elif self.unit_type == "rnn":
            rnn_type_class = torch.nn.RNN
        else:
            pass #raise error

        self.rnn = rnn_type_class(
                input_size=self.feature_size,
                hidden_size=self.layer_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=self.bidirectional)

        self.fc = torch.nn.Sequential(
            TemporalAttentionPool(self.layer_size, "key") if self.temporal_pool else nn.Identity(),
            nn.BatchNorm1d(self.layer_size) if self.batchnorm else nn.Identity(),
        )

    def forward(self, x):
        rnn_out, (h_n, c_n) = self.rnn(x)
        if not self.temporal_pool:
            rnn_out = rnn_out[:, -1] # only consider output of last time step-- what about attention-aggregation
        return {"rep": self.fc(rnn_out)}

    def get_output_size(self):
        return self.layer_size
