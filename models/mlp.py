import torch.nn as nn


class MLP(nn.Module):
    # simple multi layer perceptron, 1 hidden layer (feed forward)
    def __init__(self, config):
        super().__init__()

        # 4 * n_embd grows computation
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)

        # type of activation func like leaky relu
        self.gelu = nn.GELU(approximate="tanh")

        # projection layer, back to n_embd
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

        # dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
