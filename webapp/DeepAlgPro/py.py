# In a quick Python script or notebook
from model import convATTnet
import torch

model = convATTnet()
torch.save(model.state_dict(), "model.pt")  # overwrite old one
