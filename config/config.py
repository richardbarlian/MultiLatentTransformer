from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 512  # max sequence length
    vocab_size: int = (
        50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    )
    n_layer: int = 8  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension
    dropout: int = 0.2  # number of neurons to DROPOUT
    q_latent_dim: int = 64  # compress query vector from 784 to 64
    kv_latent_dim: int = 64  # reduce kv storage
