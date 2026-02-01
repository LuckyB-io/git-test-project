import torch

print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS device name:", torch.backends.mps.current_allocated_memory() if torch.backends.mps.is_available() else "N/A")
