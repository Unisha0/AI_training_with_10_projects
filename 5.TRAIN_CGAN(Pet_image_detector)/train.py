import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.cgan import Generator, Discriminator
from utils import get_dataloader, make_label_tensor, save_generated_images
import os

DATA_DIR = "data"
EPOCHS = 50
BATCH_SIZE = 64
IMAGE_SIZE = 64
NOISE_DIM = 100
LR = 0.0002
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader, class_names = get_dataloader(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)
num_classes = len(class_names)
img_shape = (3, IMAGE_SIZE, IMAGE_SIZE)

G = Generator(NOISE_DIM, num_classes, img_shape).to(DEVICE)
D = Discriminator(num_classes, img_shape).to(DEVICE)

criterion = nn.BCELoss()
g_optimizer = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
d_optimizer = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

for epoch in range(1, EPOCHS + 1):
    for real_imgs, labels in tqdm(loader):
        real_imgs = real_imgs.to(DEVICE)
        one_hot_labels = make_label_tensor([class_names[i] for i in labels], class_names, DEVICE)

        valid = torch.ones(real_imgs.size(0), 1).to(DEVICE)
        fake = torch.zeros(real_imgs.size(0), 1).to(DEVICE)

        # Train Generator
        g_optimizer.zero_grad()
        z = torch.randn(real_imgs.size(0), NOISE_DIM).to(DEVICE)
        gen_imgs = G(z, one_hot_labels)
        g_loss = criterion(D(gen_imgs, one_hot_labels), valid)
        g_loss.backward()
        g_optimizer.step()

        # Train Discriminator
        d_optimizer.zero_grad()
        real_loss = criterion(D(real_imgs, one_hot_labels), valid)
        fake_loss = criterion(D(gen_imgs.detach(), one_hot_labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        d_optimizer.step()

    print(f"Epoch {epoch}/{EPOCHS} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
    save_generated_images(gen_imgs, class_names[labels[0]], epoch)
    torch.save(G.state_dict(), f"models/generator_epoch_{epoch}.pt")