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
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(4,4), stride=(2,2), padding=(1,1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4,4), stride=(2,2), padding=(1,1))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(4,4), stride=(2,2), padding=(1,1))
        self.convlast = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.convlast(x)
        return x.squeeze(1)

# Combined CNN + Transformer Model
class Song2Vec(nn.Module):
    def __init__(self):
        super().__init__()

        self.batch_norm = nn.BatchNorm2d(3)  # just learn the mean and std instead of computing them from data

        self.cnn = CNNFeatureExtractor()
        
        self.gru = nn.GRU(input_size=128, hidden_size=512, num_layers=4, batch_first=True)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # B, H, W, C -> B, C, H, W

        x = self.batch_norm(x)

        x = self.cnn(x)  # B, C, H, W -> B, D, L
        
        B, D, L = x.shape

        x = x.permute(0, 2, 1) # B, D, L -> B, L, D

        _, h_n = self.gru(x)
                
        return F.normalize(h_n[-1], p=2, dim=1)

if __name__ == '__main__':
    model = Song2Vec()
    print(model)
    x = torch.randn(2, 128, 256, 3)
    y = model(x)
    print(y.shape)