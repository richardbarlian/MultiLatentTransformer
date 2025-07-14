# 🧠 Tiny Multi-Latent Transformer

> A self-driven exploration into generative modeling, inspired by the _Multi-Latent Transformer_ paper and the internals of GPT-style transformers.  
> Built entirely from scratch using **PyTorch** and **PyTorch Lightning**.

---

## 🚀 Overview

This project is a **hobby deep learning initiative** where I implemented and trained a decoder-only transformer with **modern research-backed improvements**. Inspired by the [_TransMLA: Multi-Head Latent Attention Is All You Need_ (2025)](https://arxiv.org/abs/2502.07864), it serves as a minimal and modular playground for token-level language modeling and advanced attention designs.

The project was born out of a desire to truly understand AI systems — not by using black-box APIs, but by reconstructing them from first principles.

---

## 📌 Key Features & Contributions

### ✅ Research-Inspired Design

- Built around the **Multi-Latent Transformer** concept — enabling multiple latent attention pathways for parallel reasoning.
- Closely follows the "Attention is All You Need" paper, but adopts a **post-LayerNorm** design for better stability.

### 🔧 Engineering Enhancements I Added

- ⚡️ **Memory-efficient data loading** via `PackedDataset` with overlapping (stride-based) token sequences.
- 🧠 **Cosine annealing learning rate schedule** with linear warm-up, implemented using PyTorch's `LambdaLR`.
- 🧩 **GPT-style parameter initialization** and grouped weight decay for stable convergence.
- 📈 Integrated with **TensorBoard** for simple, real-time logging of training curves and metrics.
- 🧪 **Validation perplexity evaluation** at regular intervals and end of training.
- 🛑 **Early stopping and model checkpointing** for fail-safe experiments.
- 🕸️ Created `scraper.py` to download and preprocess public domain books from **Project Gutenberg**.

---

## 🛠️ Tech Stack

- `PyTorch` + `PyTorch Lightning`
- `tiktoken` for GPT-2 style BPE tokenization
- Mixed-precision (`fp16`) training
- TensorBoard + Rich CLI logging
- Modular config-driven experiment management

---

## 📊 Training Setup

| Setting             | Value           |
| ------------------- | --------------- |
| Block Size          | 512 tokens      |
| Batch Size          | 20              |
| Max Iterations      | 18,370          |
| Learning Rate       | 3e-4            |
| Min Learning Rate   | 3e-5            |
| Warmup Steps        | 10% of total    |
| Validation Interval | every 500 steps |
| Gradient Clipping   | 1.0             |
| Accumulated Batches | 4               |
| Final Metric        | Perplexity      |

---

## ▶️ How to Use

```bash
# Clone the repo and install dependencies
pip install -r requirements.txt

# Download dataset from Project Gutenberg
python scraper.py

# Train the model
python train.py
```
