import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_model import Song2Vec

class GenreClassifier(nn.Module):
    def __init__(self, model: Song2Vec, num_genres):
        super(GenreClassifier, self).__init__()
        self.model = model
        # freeze the model
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.fc = nn.Linear(256, num_genres)

    def forward(self, x):
        x, _, _ = self.model.encode(x)
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    model = GenreClassifier(Song2Vec(), num_genres=8)
    x = torch.randn(4, 1024, 2048, 3)
    out = model(x)
    print(out.shape)