import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.convlast = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.bnlast = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        x = self.bn1(F.leaky_relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(F.leaky_relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.bn3(F.leaky_relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.bnlast(self.convlast(x))
        x = x.squeeze(1)
        return x

class CNNDecoder(nn.Module):
    def __init__(self):
        super(CNNDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(16)
        self.convlast = nn.ConvTranspose2d(16, 3, kernel_size=1, stride=1, padding=0)
        self.bnlast = nn.BatchNorm2d(3)
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.bn1(F.leaky_relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(F.leaky_relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.bn3(F.leaky_relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.bnlast(self.convlast(x))
        return x

class Song2Vec(nn.Module):
    def __init__(self):
        super(Song2Vec, self).__init__()

        self.encoder = CNNEncoder()
        self.gru_encoder = nn.GRU(
            input_size=256,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.3  # Dropout between GRU layers
        )
        
        self.enc_mapping = nn.Linear(512, 256)

        self.gru_decoder = nn.GRU(
            input_size=1,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.3  # Dropout between GRU layers
        )
        self.decoder = CNNDecoder()
        
    def encode(self, x):
        x = x.permute(0, 3, 1, 2)  # Shape: (B, C, H, W)
        x = self.encoder(x)        # Shape: (B, H_enc, W_enc)
        
        B, H_enc, W_enc = x.shape
        x = x.permute(0, 2, 1).contiguous()  # Shape: (B, W_enc, H_enc)
        
        _, h_n = self.gru_encoder(x)  # h_n: (num_layers, B, hidden_size)

        z = self.enc_mapping(h_n[-1])
        return z, h_n, W_enc  # Return z and sequence length for decoder
    
    def decode(self, h_n, seq_len):        
        # Prepare decoder inputs (zeros)
        decoder_inputs = torch.zeros(h_n.shape[1], seq_len, 1).to(h_n.device)  # Input size is 1
        
        # Run decoder GRU
        out, _ = self.gru_decoder(decoder_inputs, h_n)  # out: (B, seq_len, hidden_size)
        x_reconstructed = self.decoder(out)  # Adjust dimensions for CNNDecoder
        return x_reconstructed
    
    def forward(self, x):
        z, h_n, seq_len = self.encode(x)
        x_reconstructed = self.decode(h_n, seq_len)
        x_reconstructed = x_reconstructed.permute(0, 2, 3, 1)  # Shape: (B, H, W, C)
        return x_reconstructed, z

if __name__ == "__main__":
    model = Song2Vec()
    x = torch.randn(1, 1024, 2048, 3)  # Example input
    x_reconstructed, z = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {x_reconstructed.shape}")
    print(f"Latent vector shape: {z.shape}")
