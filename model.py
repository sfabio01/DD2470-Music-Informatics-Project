import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from fma_dataset import FmaDataset
# CNN Feature Extractor

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3,4), stride=(1,2), padding=(1,1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,4), stride=(1,2), padding=(1,1))
        self.convlast = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.convlast(x)
        return x.squeeze(1)

# Combined CNN + Transformer Model
class Song2Vec(nn.Module):
    def __init__(self):
        super().__init__()

        self.batch_norm = nn.BatchNorm2d(3)  # just learn the mean and std instead of computing them from data

        self.cnn = CNNFeatureExtractor()
        self.cls_embed = nn.Parameter(torch.empty(1024), requires_grad=True)
        self.w_pe = nn.Embedding(512, 1024)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=1024, nhead=8, dim_feedforward=2048, batch_first=True),
            num_layers=6
        )
        
        nn.init.normal_(self.cls_embed, std=0.02)

    def forward(self, x):
        DEVICE = x.device

        x = self.batch_norm(x)

        x = self.cnn(x)  # Pass through the CNN
        
        B, L, D = x.shape
        
        x = x.permute(0, 2, 1)
        
        pe = torch.arange(0, 512, device=DEVICE)
        x = torch.cat([x[:, :-1], self.cls_embed.repeat(B, 1, 1)], dim=1)
        x = x + self.w_pe(pe)
        x = self.transformer(x)[:, -1, :]  # Pass through the Transformer and extract cls
        return F.normalize(x, p=2, dim=1)  # Normalize embeddings to unit length
