import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
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
        self.cnn = CNNFeatureExtractor()
        self.cls_embed = nn.Parameter(torch.empty(1024), requires_grad=True)
        self.w_pe = nn.Embedding(512, 1024)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=1024, nhead=8, dim_feedforward=2048),
            num_layers=6
        )
        
        nn.init.normal_(self.cls_embed, std=0.02)

    def forward(self, x):
        x = self.cnn(x)  # Pass through the CNN
        
        B, L, D = x.shape
        
        x = x.permute(0, 2, 1)
        
        pe = torch.arange(0, 512)
        x = torch.cat([x[:, :-1], self.cls_embed.repeat(B, 1, 1)], dim=1)
        x = x + self.w_pe(pe)
        x = self.transformer(x)[:, -1, :]  # Pass through the Transformer and extract cls
        return F.normalize(x, p=2, dim=1)  # Normalize embeddings to unit length

if __name__ == '__main__':
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
            anchor, positive, negative = anchor.float(), positive.float(), negative.float() # shape (batch_size, 3, 1024, 2048)
            
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
