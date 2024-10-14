import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from fma_dataset import FmaDataset
from torchvision import transforms
# CNN Feature Extractor
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3,4), stride=(1,2), padding=(1,1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,4), stride=(1,2), padding=(1,1))
        self.convlast = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.convlast(x)
        return x.squeeze(1)
        

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
        x = self.transformer(x)  # Pass through the Transformer
        return x

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

    # Define the transformations
    # transform = transforms.Compose([
    #     transforms.Normalize(),  # Normalize with mean and std
    #     transforms.ToTensor()  # Convert to tensor
    # ])

    # Initialize the dataset with transformations
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
            anchor, positive, negative = torch.randn(1, 3, 1024, 2048), torch.randn(1, 4, 3, 1024, 2048), torch.randn(1, 4, 3, 1024, 2048)
        
            shape = positive.shape

            # reshape positive and negative to (batch_size * 20, n_channels, height, width)
            positive = positive.view(-1, *positive.shape[2:])
            negative = negative.view(-1, *negative.shape[2:])

            anchor_embed = model(anchor)
            with torch.no_grad():
                positive_embed = model(positive) # shape (batch_size * 20, embed_dim)
                negative_embed = model(negative)

            print("embedding shape: ", positive_embed.shape)

            # reshape positive_embed and negative_embed to (batch_size, 20, embed_dim)
            positive_embed = positive_embed.view(shape[0], shape[1], -1)
            negative_embed = negative_embed.view(shape[0], shape[1], -1)

            dist_pos = F.cosine_similarity(anchor_embed.unsqueeze(1), positive_embed, dim=-1) # shape (batch_size, 20)
            dist_neg = F.cosine_similarity(anchor_embed.unsqueeze(1), negative_embed, dim=-1)

            p = torch.argmin(dist_pos, dim=1) # shape (batch_size,)
            n = torch.argmax(dist_neg, dim=1)

            # reshape positive and negative back to (batch_size, 20, n_channels, height, width)
            positive = positive.view(shape[0], shape[1], *positive.shape[1:])
            negative = negative.view(shape[0], shape[1], *negative.shape[1:])

            # recompute positive and negative embeddings with gradients
            positive_embed = model(positive[torch.arange(shape[0]), p]) # shape (batch_size, embed_dim)
            negative_embed = model(negative[torch.arange(shape[0]), n])

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
