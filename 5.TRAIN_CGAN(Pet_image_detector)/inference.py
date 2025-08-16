import torch
from models.cgan import Generator
from utils import make_label_tensor
from torchvision.utils import save_image
import os

NOISE_DIM = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_pet(label, class_names, model_path, save_path="generated/sample.png"):
    img_shape = (3, 64, 64)
    G = Generator(NOISE_DIM, len(class_names), img_shape).to(DEVICE)
    G.load_state_dict(torch.load(model_path, map_location=DEVICE))
    G.eval()

    noise = torch.randn(1, NOISE_DIM).to(DEVICE)
    label_tensor = make_label_tensor([label], class_names, DEVICE)

    with torch.no_grad():
        generated_img = G(noise, label_tensor)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_image(generated_img, save_path, normalize=True)

    return save_path