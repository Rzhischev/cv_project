import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SELU()
            )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=2, bias=False),
            nn.BatchNorm2d(64),
            nn.SELU()
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )

        self.pool = nn.MaxPool2d(2, 2, return_indices=True, ceil_mode=True) #<<<<<< Bottleneck

        #decoder

        self.unpool = nn.MaxUnpool2d(2, 2)

        self.conv1_t = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, bias=False),
            nn.BatchNorm2d(64),
            nn.SELU()
            )

        self.conv2_t = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=2, bias=False),
            nn.BatchNorm2d(16),
            nn.SELU()
            )
        self.conv3_t = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1),
            nn.LazyBatchNorm2d(),
            nn.Sigmoid()
            )
        self.drop_out = nn.Dropout(0.5)
    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.drop_out(x)
        x, indicies = self.pool(x) # ⟸ bottleneck
        return x, indicies

    def decode(self, x, indicies):
        x = self.unpool(x, indicies)
        x = self.conv1_t(x)
        x = self.conv2_t(x)
        x = self.drop_out(x)
        x = self.conv3_t(x)
        return x

    def forward(self, x):
        latent, indicies = self.encode(x)
        out = self.decode(latent, indicies)
        return out
