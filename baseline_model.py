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
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.convlast(x)  # Shape: (B, 1, H, W)
        x = x.squeeze(1)      # Shape: (B, H, W)
        return x

class CNNDecoder(nn.Module):
    def __init__(self):
        super(CNNDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(16)
        self.convlast = nn.ConvTranspose2d(16, 3, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Shape: (B, 1, H, W)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.convlast(x)  # Shape: (B, 3, H, W)
        return x

class Song2Vec(nn.Module):
    def __init__(self):
        super(Song2Vec, self).__init__()

        self.encoder = CNNEncoder()
        self.gru_encoder = nn.GRU(input_size=256, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)

        self.hid_mapping = nn.Linear(512 * 2 * 2, 256 * 2 * 2)  # Mapping from encoder to decoder hidden state

        self.gru_decoder = nn.GRU(input_size=1, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        self.fc_dec = nn.Linear(in_features=512, out_features=256)
        self.decoder = CNNDecoder()
        
    def encode(self, x):
        x = x.permute(0, 3, 1, 2)  # Shape: (B, C, H, W)
        x = self.encoder(x)        # Shape: (B, H_enc, W_enc)
        
        B, H_enc, W_enc = x.shape
        x = x.permute(0, 2, 1).contiguous()  # Shape: (B, W_enc, H_enc)
        
        out, h_n = self.gru_encoder(x)  # h_n: (num_layers * num_directions, B, hidden_size)
        z = h_n  # We'll use the entire hidden state
        z = z.permute(1, 0, 2).contiguous().view(B, -1)
        return z, W_enc  # Return z and sequence length for decoder
    
    def decode(self, z, seq_len):
        # Map encoder hidden state z to decoder initial hidden state h_0
        h_0 = self.hid_mapping(z)  # Shape: (B, num_layers * num_directions * decoder_hidden_size)
        h_0 = h_0.view(z.shape[0], self.gru_decoder.num_layers * (2 if self.gru_decoder.bidirectional else 1), self.gru_decoder.hidden_size)
        h_0 = h_0.permute(1, 0, 2).contiguous()  # Shape: (num_layers * num_directions, B, decoder_hidden_size)
        
        # Prepare decoder inputs (zeros)
        decoder_inputs = torch.zeros(z.shape[0], seq_len, 1).to(z.device)  # Input size is 1
        
        # Run decoder GRU
        out, _ = self.gru_decoder(decoder_inputs, h_0)  # out: (B, seq_len, num_directions * hidden_size)
        out = self.fc_dec(out)  # Shape: (B, seq_len, 256)
        x_reconstructed = self.decoder(out.permute(0, 2, 1))  # Adjust dimensions for CNNDecoder
        return x_reconstructed
    
    def forward(self, x):
        z, seq_len = self.encode(x)
        x_reconstructed = self.decode(z, seq_len)
        x_reconstructed = x_reconstructed.permute(0, 2, 3, 1)  # Shape: (B, H, W, C)
        return x_reconstructed, z

if __name__ == "__main__":
    model = Song2Vec()
    x = torch.randn(1, 1024, 2048, 3)  # Example input
    x_reconstructed, z = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {x_reconstructed.shape}")
    print(f"Latent vector shape: {z.shape}")
