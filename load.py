import torch
import torchvision
from model import build_model
from utils import save_model_all


# Constants.
DATA_PATH = "../input/test_images"
IMAGE_SIZE = 224
DEVICE = "cpu"
# Class names.
class_names = ["Bishop", "King", "Knight", "Pawn", "Queen", "Rook"]
# Load the trained model.
model = build_model(pretrained=False, fine_tune=False, num_classes=5)
checkpoint = torch.load("outputs/model_pretrained_True.pth", map_location=DEVICE)
print("Loading trained model weights...")
model.load_state_dict(checkpoint["model_state_dict"])

model_dynamic_quantized = torch.quantization.quantize_dynamic(
    model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
)

save_model_all(model_dynamic_quantized, "quantized")


backend = "qnnpack"
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend
model_static_quantized = torch.quantization.prepare(model, inplace=False)
model_static_quantized = torch.quantization.convert(
    model_static_quantized, inplace=False
)

save_model_all(model_dynamic_quantized, "static_quantized")
