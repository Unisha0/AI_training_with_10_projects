# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

from utils import save_generated_images

def get_mnist_dataset(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model_type, generator, discriminator, z_dim, device, epochs=10, batch_size=128):
    os.makedirs("generated", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    dataloader = get_mnist_dataset(batch_size)
    loss_fn = nn.BCELoss()
    g_opt = optim.Adam(generator.parameters(), lr=0.0002)
    d_opt = optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(epochs):
        for real, _ in dataloader:
            real = real.to(device)
            batch_size = real.size(0)

            # Sample random noise
            if model_type == "DCGAN":
                noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            else:
                noise = torch.randn(batch_size, z_dim).to(device)

            # Fake data
            fake = generator(noise if model_type == "DCGAN" else noise.view(batch_size, -1))

            # Discriminator training
            d_real = discriminator(real).view(-1)
            d_fake = discriminator(fake.detach()).view(-1)
            d_loss = loss_fn(d_real, torch.ones_like(d_real)) + loss_fn(d_fake, torch.zeros_like(d_fake))

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # Generator training
            output = discriminator(fake).view(-1)
            g_loss = loss_fn(output, torch.ones_like(output))

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

        # Save image after each epoch
        save_generated_images(generator, z_dim, epoch+1, model_type, device)

    # Save final models
    torch.save(generator.state_dict(), f"models/{model_type.lower()}_gen.pt")
    torch.save(discriminator.state_dict(), f"models/{model_type.lower()}_disc.pt")