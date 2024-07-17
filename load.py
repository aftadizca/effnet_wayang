import torch
import cv2
import numpy as np
import glob as glob
import os
from model import build_model
from torchvision import transforms
from utils import save_model_all

# Constants.
DATA_PATH = "../input/test_images"
IMAGE_SIZE = 224
DEVICE = "cpu"
# Class names.
class_names = ["Bishop", "King", "Knight", "Pawn", "Queen", "Rook"]
# Load the trained model.
model = build_model(pretrained=False, fine_tune=False, num_classes=5)
checkpoint = torch.load("../outputs/model_pretrained_True.pth", map_location=DEVICE)
print("Loading trained model weights...")
model.load_state_dict(checkpoint["model_state_dict"])

save_model_all(model=model)
