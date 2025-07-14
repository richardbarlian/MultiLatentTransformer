import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from config.config import GPTConfig
from models.gpt import GPT
import tiktoken
import math
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# hyperparameters
block_size = 512
batch_size = 20
max_iters = 18370
eval_interval = 500
learning_rate = 3e-4
checkpoint_iterations = 500
warmup_steps = max_iters // 10

torch.set_float32_matmul_precision("high")

# load dataset
with open("data/dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
vocab_size = tokenizer.n_vocab

split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]


class PackedDataset(Dataset):
    def __init__(self, data, block_size, stride=None):
        self.block_size = block_size
        self.stride = stride or block_size  # full block if no overlap

        self.samples = []
        for i in range(0, len(data) - block_size, self.stride):
            self.samples.append(data[i : i + block_size])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        x = torch.tensor(x, dtype=torch.long)
        y = torch.roll(x, shifts=-1)  # next-token prediction
        return x, y


# data loaders
train_dataset = PackedDataset(train_data.tolist(), block_size, stride=block_size // 2)
val_dataset = PackedDataset(val_data.tolist(), block_size, stride=block_size // 2)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=False,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=False,
    persistent_workers=True,
)

# model setup
config = GPTConfig(vocab_size=vocab_size)


class GPTLightningModule(pl.LightningModule):
    def __init__(self, config, learning_rate, min_lr, warmup_steps, max_iters):
        super().__init__()
        self.model = GPT(config, use_checkpoint=True)
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_iters = max_iters
        self.save_hyperparameters()
        self.apply_gpt_init()

    def forward(self, x, y):
        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self(x, y)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("lr", lr, on_step=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self(x, y)
        val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("bias") or "ln" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": 1e-2},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
        )

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(
                max(1, self.max_iters - self.warmup_steps)
            )
            return max(
                self.min_lr / self.learning_rate,
                0.5 * (1.0 + math.cos(math.pi * progress)),
            )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "interval": "step",
        }

        return [optimizer], [scheduler]

    def apply_gpt_init(self):
        def init_fn(module):
            if isinstance(module, torch.nn.Linear):
                init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                init.normal_(module.weight, mean=0.0, std=0.02)

        self.model.apply(init_fn)


# initialize model
model = GPTLightningModule(
    config=config,
    learning_rate=learning_rate,
    min_lr=3e-5,
    warmup_steps=warmup_steps,
    max_iters=max_iters,
)

# logger and checkpointing
logger = TensorBoardLogger("logs/", name="gpt_model")

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="checkpoint-{step}",
    save_top_k=-1,
    every_n_train_steps=checkpoint_iterations,
    monitor="val_loss",
    mode="min",
)

early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.0,
    patience=10,
    mode="min",
    check_on_train_epoch_end=False,
    strict=True,
)

# trainer
trainer = pl.Trainer(
    max_steps=max_iters,
    max_epochs=10,
    precision="16-mixed",
    logger=logger,
    log_every_n_steps=50,
    val_check_interval=eval_interval,
    callbacks=[checkpoint_callback, RichProgressBar(), early_stopping_callback],
    accelerator="auto",
    devices=1,
    limit_val_batches=700,
    accumulate_grad_batches=4,
    gradient_clip_val=1.0,
)

if __name__ == "__main__":
    trainer.fit(model, train_loader, val_loader)

    # final evaluation
    model.eval()
    with torch.no_grad():
        xb, yb = next(iter(val_loader))
        logits, _ = model(xb, yb)
        final_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        final_loss = final_loss.item()
        perplexity = torch.exp(torch.tensor(final_loss)).item()
        print(f"Final validation loss: {final_loss:.4f}")
        print(f"Final validation perplexity: {perplexity:.2f}")

    torch.save(model.state_dict(), "checkpoints/gpt_trained.pth")
    print("Saved final model to checkpoints/gpt_trained.pth")
