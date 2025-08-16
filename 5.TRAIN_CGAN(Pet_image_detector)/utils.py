import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.nn.functional import one_hot
import torch

def get_dataloader(data_dir, image_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = ImageFolder(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), dataset.classes


def make_label_tensor(labels, class_names, device):
    label_idx = [class_names.index(l) for l in labels]
    return one_hot(torch.tensor(label_idx), num_classes=len(class_names)).float().to(device)


def save_generated_images(images, label_name, epoch):
    os.makedirs(f"generated/{label_name}", exist_ok=True)
    save_image(images.data[:8], f"generated/{label_name}/epoch_{epoch}.png", nrow=4, normalize=True)