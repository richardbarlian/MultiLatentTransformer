import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from models.block import Block


class GPT(nn.Module):
    def __init__(self, config, use_checkpoint=False):
        super().__init__()
        self.config = config
        self.use_checkpoint = use_checkpoint

        self.transformer = nn.ModuleDict(
            dict(
                # token embedding
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # positional embedding
                wpe=nn.Embedding(config.block_size, config.n_embd),
                # layers of blocks
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                # final layer norm
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        # fully connected
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # batch size, block size
        B, T = idx.shape

        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # token embeddings is learned of n_embd size
        tok_emb = self.transformer.wte(idx)  # (B,T,C)
        # positional embeddings is learned by the position
        pos_emb = self.transformer.wpe(torch.arange(T, device=idx.device))  # (T, C)

        x = tok_emb + pos_emb

        # forward the blocks of the transformer with optional checkpointing
        for block in self.transformer.h:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x)
            else:
                x = block(x)

        # layer norm
        x = self.transformer.ln_f(x)  # (B, T, vocab_size)

        # each embedding is fully connected to a layer of vocab_size shape
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # logits.view(-1, logits.size(-1)) squashes from (B, T, vocab_size)
            # to (B*T, vocab_size)
            # to (B*T*vocab_size, ), which can be used for cross entropy with (B, T) to (B*T, )
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
