import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))  # i.e. half thrice twice, freq twice
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.convlast = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.convlast(x)  # Shape: (B, 1, H, W)
        x = x.squeeze(1)      # Shape: (B, H, W)
        return x

class Song2Vec(nn.Module):
    def __init__(self):
        super(Song2Vec, self).__init__()

        self.encoder = CNNEncoder()
        self.gru_encoder = nn.GRU(input_size=256, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.encoder(x)
        
        B, H_enc, W_enc = x.shape
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, W_enc, -1)
        
        _, h_n = self.gru_encoder(x)
        z = h_n[-1]
        z = F.normalize(z, p=2, dim=1)

        return z

if __name__ == "__main__":
    model = Song2Vec()
    x = torch.randn(1, 1024, 2048, 3)  # Example input
    z = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Latent vector shape: {z.shape}")
