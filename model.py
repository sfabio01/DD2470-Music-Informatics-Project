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
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.convlast(x))
        return x.squeeze(1)
    
class CNNDecoder(nn.Module):
    def __init__(self):
        super(CNNDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(1, 32, kernel_size=(4,3), stride=(2,1), padding=(1,1))
        self.conv2 = nn.ConvTranspose2d(32, 16, kernel_size=(4,3), stride=(2,1), padding=(1,1))
        self.convlast = nn.ConvTranspose2d(16, 3, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        return self.convlast(x)
    

class Song2Vec(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = CNNEncoder()

        self.w_pe = nn.Embedding(512, 1024)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=1024, nhead=8, dim_feedforward=2048, batch_first=True),
            num_layers=3
        )
        
        self.query_vector = nn.Parameter(torch.randn(1, 1, 1024))
        
        self.transformer_decoder = nn.TransformerEncoder(  # decoder in the sense that it is after the middle
            nn.TransformerEncoderLayer(d_model=1024, nhead=8, dim_feedforward=2048, batch_first=True),
            num_layers=3
        )

        self.decoder = CNNDecoder()
    
    def encode(self, x):
        DEVICE = x.device

        x = x.permute(0, 3, 2, 1)

        x = self.encoder(x)
        B, L, D = x.shape
        pos = self.w_pe(torch.arange(L, device=DEVICE)).unsqueeze(0).expand(B, -1, -1)
        context = x + pos
        context_w_query = torch.cat([context, self.query_vector.expand(B, -1, -1)], dim=1)
        context_w_query = self.transformer_encoder(context_w_query)

        context = context_w_query[:, :-1, :]
        
        z = context_w_query[:, -1, :]   
        z = F.normalize(z, p=2, dim=1)
            
        return context, z

    def decode(self, context):
        x = self.transformer_decoder(context)
        x = self.decoder(x)
        return x

    def forward(self, x):
        x, z = self.encode(x)
        x = self.decode(x)
        return x.permute(0, 3, 2, 1), z
    

if __name__ == "__main__":
    model = Song2Vec()
    x = torch.randn(1, 1024, 2048, 3)
    x, z = model(x)
    print(x.shape, z.shape)