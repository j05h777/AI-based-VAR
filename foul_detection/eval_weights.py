import torch
from pathlib import Path

BASE = Path(__file__).resolve().parent
ckpt_path = BASE / "14_model.pth.tar"

checkpoint = torch.load(ckpt_path, map_location="cpu")
print(checkpoint.keys())
print(checkpoint["epoch"])