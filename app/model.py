from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "resnet18_car.pth"


def build_model():
	model = models.resnet18(weights=None)
	model.fc = nn.Linear(model.fc.in_features, 2)
	return model


@lru_cache(maxsize=1)
def get_model():
	if not MODEL_PATH.exists():
		raise FileNotFoundError(
			f"Model checkpoint not found at {MODEL_PATH}. Run ml/train.py to create it."
		)

	model = build_model()
	model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
	model.to(device)
	model.eval()
	return model