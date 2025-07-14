import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.head_dim = config.n_embd // config.n_head
        assert (
            self.head_dim == config.kv_latent_dim
        ), "For MLA attention, head_dim must equal kv_latent_dim"

        # MLA-style low-rank query and key/value compression layers
        self.Wq_d = nn.Linear(config.n_embd, config.q_latent_dim)
        self.W_qk = nn.Linear(config.q_latent_dim, config.n_head * config.kv_latent_dim)

        self.Wkv_d = nn.Linear(config.n_embd, config.kv_latent_dim)
        self.Wv_u = nn.Linear(config.kv_latent_dim, config.n_head * self.head_dim)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.kv_latent_dim = config.kv_latent_dim

    def forward(self, x):
        B, T, C = x.size()
        # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer

        # MLA-style attention
        C_q = self.Wq_d(x)  # (B, T, q_latent_dim)
        C_kv = self.Wkv_d(x)  # (B, T, kv_latent_dim)

        q = (
            self.W_qk(C_q).view(B, T, self.n_head, self.kv_latent_dim).transpose(1, 2)
        )  # (B, nh, T, kv_latent_dim)
        # for keys, project C_kv to (B, T, kv_latent_dim), then (B, 1, kv_latent_dim, T)
        k = C_kv.transpose(1, 2).unsqueeze(1)  # (B, 1, kv_latent_dim, T)
        # q: (B, nh, T, kv_latent_dim), k: (B, 1, kv_latent_dim, T)
        # note: head_dim and kv_latent_dim must be equal for this scaling
        scores = torch.matmul(q, k) / (self.head_dim**0.5)  # (B, nh, T, T)
        attn_weight = F.softmax(scores, dim=-1)

        # value projection and MLA-style output
        v = (
            self.Wv_u(C_kv).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        )  # (B, nh, T, head_dim)
        y = torch.matmul(attn_weight, v)  # (B, nh, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
