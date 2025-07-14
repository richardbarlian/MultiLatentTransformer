import torch.nn as nn
from models.causalselfattention import CausalSelfAttention
from models.mlp import MLP


class Block(nn.Module):
    # block: communication followed by computation
    def __init__(self, config):
        # n_embd: embedding dimension, n_head: the number of heads
        super().__init__()
        self.config = config

        # self attention multi head
        self.attn = CausalSelfAttention(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)  # layer norm for attn

        # feedforward allows further relationships between affinities
        self.mlp = MLP(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)  # layer norm for mlp

        # dropout for residuals
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x + -> residual connections
        # + means gradient can flow backwards during backprop, no vanishing gradient
        x = x + self.resid_dropout(self.attn(self.ln_1(x)))
        x = x + self.resid_dropout(self.mlp(self.ln_2(x)))
        return x
