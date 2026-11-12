"""Train a small GPT language model on TinyStories.

The same file runs on a Mac (Apple GPU), on an NVIDIA GPU, or on a CPU:
    uv run --with-requirements requirements.txt train.py     # on your machine
    ./run_on_hf.sh train.py                                  # on a Hugging Face GPU
"""

import datetime
import math
import os
import time

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

tokenizer = tiktoken.get_encoding("gpt2")

# Model: about 51M parameters, half of them in the token embedding table
VOCAB_SIZE = tokenizer.n_vocab  # 50,257 tokens
BLOCK_SIZE = 512                # context length, in tokens
N_LAYER = 8
N_HEAD = 8
N_EMBD = 512

# Training
BATCH_SIZE = 16                 # 16 sequences of 512 tokens = 8,192 tokens per step
LR = 6e-4
WARMUP_STEPS = 1000


class Block(nn.Module):
    """One transformer layer: causal self-attention, then an MLP, each added back to its input."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok_emb.weight  # the output layer reuses the token embedding table
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.blocks(self.tok_emb(tokens) + self.pos_emb(positions))
        return self.head(self.ln(x))  # scores for the next token, at every position

    @torch.no_grad()
    def generate(self, prompt, max_tokens=300, temperature=0.8):
        tokens = torch.tensor([tokenizer.encode(prompt)], device=self.head.weight.device)
        for _ in range(max_tokens):
            logits = self(tokens[:, -BLOCK_SIZE:])[0, -1]
            next_token = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)
            if next_token.item() == tokenizer.eot_token:  # end of the story
                break
            tokens = torch.cat([tokens, next_token[None]], dim=1)
        return tokenizer.decode(tokens[0].tolist())


# Training. This part runs only when the file is executed, not when sample.py imports the model.
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Download TinyStories and turn it into a single long array of tokens
    chunks = []
    for batch in load_dataset("roneneldan/TinyStories", split="train").iter(batch_size=10_000):
        for tokens in tokenizer.encode_ordinary_batch(batch["text"]):
            chunks.append(np.array(tokens + [tokenizer.eot_token], dtype=np.uint16))
    data = np.concatenate(chunks)
    max_steps = len(data) // (BATCH_SIZE * BLOCK_SIZE)  # about one pass over the dataset

    model = GPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    print(f"Device: {device} | {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters | "
          f"{len(data) / 1e6:.0f}M training tokens | {max_steps:,} steps")

    # On NVIDIA GPUs, compiling the model and computing in bfloat16 make training much faster.
    # Neither helps on a Mac. The compiled copy is used for training steps only.
    fast_model = torch.compile(model) if device == "cuda" else model
    start = time.time()

    for step in range(1, max_steps + 1):
        # Learning rate: linear warmup, then cosine decay down to 10% of the peak value
        if step < WARMUP_STEPS:
            lr = LR * step / WARMUP_STEPS
        else:
            progress = (step - WARMUP_STEPS) / (max_steps - WARMUP_STEPS)
            lr = LR * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Random windows of text. The targets are the inputs shifted by one token.
        starts = np.random.randint(0, len(data) - BLOCK_SIZE - 1, BATCH_SIZE)
        windows = np.stack([data[s : s + BLOCK_SIZE + 1] for s in starts]).astype(np.int64)
        windows = torch.from_numpy(windows).to(device)
        inputs, targets = windows[:, :-1], windows[:, 1:]

        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = fast_model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            elapsed = time.time() - start
            time_left = datetime.timedelta(seconds=round(elapsed / step * (max_steps - step)))
            tokens_per_sec = step * BATCH_SIZE * BLOCK_SIZE / elapsed
            print(f"step {step:6d}/{max_steps} | loss {loss.item():.3f} | "
                  f"{tokens_per_sec:,.0f} tokens/s | time left {time_left}")

        if step % 2000 == 0:  # show what the model writes at this point
            print(f"\n{model.generate('Once upon a time')}\n")

    os.makedirs("out", exist_ok=True)
    torch.save(model.state_dict(), "out/model.pt")
    print(f"Model saved to out/model.pt after {datetime.timedelta(seconds=round(time.time() - start))}")
