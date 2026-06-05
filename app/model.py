from functools import lru_cache
from pathlib import Path

import onnxruntime as ort

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "resnet18_car.onnx"


@lru_cache(maxsize=1)
def get_model():
	if not MODEL_PATH.exists():
		raise FileNotFoundError(
			f"ONNX model not found at {MODEL_PATH}. Export it from models/resnet18_car.pth first."
		)

	return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
