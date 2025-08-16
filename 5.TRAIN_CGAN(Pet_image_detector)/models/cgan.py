import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, label_dim, img_shape):
        super().__init__()
        input_dim = noise_dim + label_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, noise, labels):
        x = torch.cat((noise, labels), dim=1)
        out = self.model(x)
        return out.view(out.size(0), *self.img_shape)


class Discriminator(nn.Module):
    def __init__(self, label_dim, img_shape):
        super().__init__()
        input_dim = label_dim + int(torch.prod(torch.tensor(img_shape)))
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        x = torch.cat((img.view(img.size(0), -1), labels), dim=1)
        return self.model(x)