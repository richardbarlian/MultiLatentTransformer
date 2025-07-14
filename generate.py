import torch
from config.config import GPTConfig
from models.gpt import GPT
import tiktoken
from torch.nn import functional as F
import pytorch_lightning as pl
from train import GPTLightningModule  # Make sure this import path matches your project

# Device setup
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab

# Load model from .ckpt file
ckpt_path = "checkpoints/checkpoint-step=1000.ckpt"  # adjust filename as needed
model = GPTLightningModule.load_from_checkpoint(
    ckpt_path,
    config=GPTConfig(vocab_size=vocab_size),
    learning_rate=6e-4,  # match training values
    min_lr=6e-5,
    warmup_steps=2000,
    max_iters=10000,
)
model.to(device)
model.eval()


# Generation function
@torch.no_grad()
def generate(prompt, max_length=200, temperature=1.0, top_k=50):
    input_ids = torch.tensor(
        tokenizer.encode(prompt), dtype=torch.long, device=device
    ).unsqueeze(0)

    for _ in range(max_length):
        logits, _ = model.model(input_ids[:, -model.model.config.block_size :])

        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, top_k)
        next_token = torch.multinomial(top_k_probs, 1)
        next_token = top_k_indices.gather(-1, next_token)

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0].tolist())


# Example
prompt = "I hate the wind."
print(generate(prompt))
