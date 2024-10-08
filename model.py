import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from fma_dataset import FmaDataset
# CNN Feature Extractor
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, output_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        return x

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Take the mean over the time dimension for representation
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)  # Normalize embeddings to unit length

# Combined CNN + Transformer Model
class Song2Vec(nn.Module):
    def __init__(self, cnn_input_channels, cnn_output_channels, transformer_input_dim, embed_dim, num_heads, num_layers, ff_dim):
        super().__init__()
        self.cnn = CNNFeatureExtractor(cnn_input_channels, cnn_output_channels)
        self.transformer = TransformerEncoder(transformer_input_dim, embed_dim, num_heads, num_layers, ff_dim)

    def forward(self, x):
        x = self.cnn(x)  # Pass through the CNN
        x = x.permute(0, 2, 1)  # Transpose for Transformer (batch, seq_len, feature_dim)
        x = self.transformer(x)  # Pass through the Transformer
        return x

# Example model initialization
cnn_input_channels = 1  # Assuming the input is a 1D array
cnn_output_channels = 128  # Number of output channels after CNN
transformer_input_dim = cnn_output_channels  # Should match the output channels from the CNN
embed_dim = 64  # Output embedding size for transformer
num_heads = 4
num_layers = 3
ff_dim = 256

model = Song2Vec(cnn_input_channels, cnn_output_channels, transformer_input_dim, embed_dim, num_heads, num_layers, ff_dim)

# Triplet Margin Loss
triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

# Example dataset initialization
metadata_folder = 'path_to_metadata'
root_dir = 'path_to_npy_files'
dataset = FmaDataset(metadata_folder, root_dir, transform=None)

# DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (anchor, positive, negative) in enumerate(dataloader):
        anchor, positive, negative = anchor.float(), positive.float(), negative.float()

        # Add a dimension for the channel before passing through CNN (assuming 1D input data)
        anchor = anchor.unsqueeze(1)  # Shape: (batch_size, 1, seq_len)
        positive = positive.unsqueeze(1)
        negative = negative.unsqueeze(1)
        
        # Forward pass
        anchor_embed = model(anchor)
        positive_embed = model(positive)
        negative_embed = model(negative)
        
        # Compute Triplet Loss
        loss = triplet_loss_fn(anchor_embed, positive_embed, negative_embed)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 mini-batches
            print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 10:.4f}")
            running_loss = 0.0

print("Training finished.")
