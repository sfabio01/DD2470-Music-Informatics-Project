import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1))
        self.convlast = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = self.convlast(x)  # Shape: (B, 1, H, W)
        x = x.squeeze(1)      # Shape: (B, H, W)
        return x

class CNNDecoder(nn.Module):
    def __init__(self):
        super(CNNDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1))
        self.convlast = nn.ConvTranspose2d(16, 3, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Shape: (B, 1, H, W)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = self.convlast(x)  # Shape: (B, 3, H, W)
        return x

class Song2Vec(nn.Module):
    def __init__(self):
        super(Song2Vec, self).__init__()

        self.encoder = CNNEncoder()

        # GRU Encoder
        self.gru_encoder = nn.GRU(input_size=256, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)

        # GRU Decoder (512 x 2 for bidirectional)
        self.gru_decoder = nn.GRU(input_size=512, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)

        self.fc_dec = nn.Linear(in_features=512, out_features=256)
        self.decoder = CNNDecoder()
        
        
    def encode(self, x):
        # x shape: (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # Shape: (B, C, H, W)
        x = self.encoder(x)        # Shape: (B, H_enc, W_enc)
        
        B, H_enc, W_enc = x.shape
        x = x.permute(0, 2, 1)     # Shape: (B, W_enc, H_enc)
        x = x.contiguous().view(B, W_enc, -1)  # Shape: (B, W_enc, H_enc)
        
        # GRU Encoder
        out, h_n = self.gru_encoder(x)  # out shape: (B, W_enc, 1024), h_n shape: (num_layers * num_directions, B, 512)
        z = h_n[-1]  # Taking the last layer's hidden state as the latent vector, shape: (B, 512)
        z = F.normalize(z, p=2, dim=1)  # Normalize the latent vector

        return out, z

    def decode(self, context):
        # context shape: (B, W_enc, 1024)
        # GRU Decoder
        out, _ = self.gru_decoder(context)  # Shape: (B, W_enc, 1024)

        out = self.fc_dec(out)

        x = self.decoder(out.permute(0, 2, 1))  # Shape: (B, 3, H, W)

        return x

    def forward(self, x):
        context, z = self.encode(x)
        x_reconstructed = self.decode(context)
        x_reconstructed = x_reconstructed.permute(0, 2, 3, 1)  # Shape: (B, H, W, C)
        return x_reconstructed, z

if __name__ == "__main__":
    model = Song2Vec()
    x = torch.randn(1, 1024, 2048, 3)  # Example input
    x_reconstructed, z = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {x_reconstructed.shape}")
    print(f"Latent vector shape: {z.shape}")
