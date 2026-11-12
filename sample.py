"""Write stories with the trained model saved in out/model.pt.

    uv run --with-requirements requirements.txt sample.py
    uv run --with-requirements requirements.txt sample.py "The little dog"
"""

import sys

import torch

from train import GPT

model = GPT()
model.load_state_dict(torch.load("out/model.pt", map_location="cpu"))
prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
for _ in range(3):
    print(model.generate(prompt), end="\n\n")
