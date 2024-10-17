import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(4,3), stride=(2,1), padding=(1,1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4,3), stride=(2,1), padding=(1,1))
        self.convlast = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.convlast(x))
        return x.squeeze(1)
    
class CNNDecoder(nn.Module):
    def __init__(self):
        super(CNNDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(1, 32, kernel_size=(4,3), stride=(2,1), padding=(1,1))
        self.conv2 = nn.ConvTranspose2d(32, 16, kernel_size=(4,3), stride=(2,1), padding=(1,1))
        self.convlast = nn.ConvTranspose2d(16, 3, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.convlast(x)
    

class Song2Vec(nn.Module):
    def __init__(self):
        super().__init__()

        self.batch_norm = nn.BatchNorm2d(3)  # just learn the mean and std instead of computing them from data

        self.encoder = CNNEncoder()

        self.w_pe = nn.Embedding(512, 1024)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=1024, nhead=8, dim_feedforward=2048, batch_first=True),
            num_layers=3
        )
        
        # Attention layer
        # self.latent_summary = nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
        # self.downsample = nn.Linear(1024, 128)
        
        self.transformer_decoder = nn.TransformerEncoder(  # decoder in the sense that it is after the middle
            nn.TransformerEncoderLayer(d_model=1024, nhead=8, dim_feedforward=2048, batch_first=True),
            num_layers=3
        )

        self.decoder = CNNDecoder()

        self.output_scale = nn.Parameter(torch.ones(1, 3, 1, 1), requires_grad=True)  # Learnt mean and std
        self.output_shift = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=True)

    def scale(self, x):
        return x * self.output_scale + self.output_shift
    
    def encode(self, x):
        DEVICE = x.device
        
        x = self.batch_norm(x.permute(0, 3, 2, 1))

        x = self.encoder(x)
        B, L, D = x.shape
        pos = self.w_pe(torch.arange(L, device=DEVICE)).unsqueeze(0).expand(B, -1, -1)
        x = x + pos
        context = self.transformer_encoder(x)
        
        # Attention to create single embedding
        # z, _ = self.latent_summary(context.mean(dim=1, keepdim=True), context, context)
        # z = z.squeeze(1)
        # z = self.downsample(z)
        z = context.mean(dim=1)
        z = F.normalize(z, p=2, dim=1)
        
        return context, z

    def decode(self, context):
        x = self.transformer_decoder(context)
        x = self.decoder(x)
        return x

    def forward(self, x):
        x, z = self.encode(x)
        x = self.decode(x)
        return self.scale(x).permute(0, 3, 2, 1), z
    

if __name__ == "__main__":
    model = Song2Vec()
    x = torch.randn(1, 1024, 2048, 3)
    x, z = model(x)
    print(x.shape, z.shape)