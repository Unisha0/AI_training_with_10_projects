# utils.py

import torch
from torchvision.utils import save_image
import os

def save_generated_images(generator, z_dim, epoch, model_type, device):
    generator.eval()
    with torch.no_grad():
        if model_type == "DCGAN":
            noise = torch.randn(64, z_dim, 1, 1).to(device)
        else:
            noise = torch.randn(64, z_dim).to(device)
        fake_images = generator(noise if model_type == "DCGAN" else noise.view(64, -1))
        save_image(fake_images, f"generated/{model_type.lower()}_output_epoch{epoch}.png", normalize=True)
    generator.train()