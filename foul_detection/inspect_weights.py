import torch
from pathlib import Path

BASE = Path(__file__).resolve().parent
path = BASE / "14_model.pth.tar"

print("PATH:", path)
print("EXISTS:", path.exists())

ckpt = torch.load(path, map_location="cpu")
print(type(ckpt))

if isinstance(ckpt, dict):
    print(ckpt.keys())