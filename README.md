# 🧠 Tiny Multi-Latent Transformer

- A self-driven exploration into generative modeling
- Inspired by the _Multi-Latent Transformer_ paper and GPT-style transformer internals
- Built entirely from scratch using **PyTorch** and **PyTorch Lightning**

## Overview

- Hobby deep learning initiative implementing a **decoder-only transformer**
- Incorporates modern research-backed improvements
- Serves as a minimal, modular playground for token-level language modeling
- Goal was to understand AI systems by reconstructing them from first principles

## Key Features & Contributions

### Research-Inspired Design

- Built on the **Multi-Latent Transformer** concept for parallel reasoning via multiple latent attention pathways
- Adopts a **post-LayerNorm** design for better stability, following the "Attention is All You Need" structure

### Engineering Enhancements

- **Memory-efficient data loading** via `PackedDataset` with overlapping token sequences
- **Cosine annealing learning rate schedule** with linear warm-up
- **GPT-style parameter initialization** and grouped weight decay for stable convergence
- **Integrated with TensorBoard** for real-time logging of training curves and metrics
- **Validation perplexity evaluation** at regular intervals
- **Early stopping and model checkpointing** for fail-safe experiments
- Created `scraper.py` for downloading and preprocessing public domain books from **Project Gutenberg**

## Technical Stack 🛠️

- `PyTorch` + `PyTorch Lightning`
- `tiktoken` for GPT-2 style BPE tokenization
- Mixed-precision (`fp16`) training
- TensorBoard + Rich CLI logging
- Modular config-driven experiment management

## Training Setup

| Setting             | Value           |
| :------------------ | :-------------- |
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

## How to Use

```bash
# Clone the repo and install dependencies (preferably in a virtual env like Anaconda)
pip install -r requirements.txt

# Download dataset from Project Gutenberg
python scraper.py

# Train the model
python train.py
```
