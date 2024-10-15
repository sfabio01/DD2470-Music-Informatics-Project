import argparse
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, random_split

import wandb
import numpy as np
from tqdm import tqdm

torch.random.manual_seed(1337)

from model import Song2Vec
from fma_dataset import FmaDataset

def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE=="cuda" else torch.float16

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"Training on device: {DEVICE}")

    train_ds = FmaDataset(metadata_folder="fma_metadata", root_dir="fma_processed", split="train")
    val_ds = FmaDataset(metadata_folder="fma_metadata", root_dir="fma_processed", split="val") #TODO : Normalise and Standaridise
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, prefetch_factor=args.prefetch_factor, num_workers=args.num_workers, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, prefetch_factor=args.prefetch_factor, num_workers=args.num_workers, persistent_workers=True)


    model = Song2Vec().to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.9)
    
    print(f"training model with {sum([p.numel() for p in model.parameters() if p.requires_grad])/1e6:.2f}M parameters")

    wandb.init(
        name=args.run_name,
        project="Song2Vec",
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "n_training_examples": len(train_ds),
            "n_validation_examples": len(val_ds),
            "parameter_count": sum([p.numel() for p in model.parameters() if p.requires_grad]),
            **vars(args)
        },
    )

        
    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

    model = torch.compile(model, backend="aot_eager")
    model.train()

    for epoch in range(args.epochs):
        train_tqdm = tqdm(train_dl, desc=f"Epoch {epoch + 1}/{args.epochs} Training")

        for i, (anchor, positive, negative) in enumerate(train_tqdm):
            anchor, positive, negative = anchor.float(), positive.float(), negative.float() # shape (batch_size, 3, 1024, 2048)
            
            with torch.autocast(device_type=DEVICE, dtype=DTYPE):
                anchor_embed = model(anchor)
                positive_embed = model(positive)
                negative_embed = model(negative)
            
                # Compute Triplet Loss
                loss = triplet_loss_fn(anchor_embed, positive_embed, negative_embed)
            
            # Backward pass
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            running_loss += loss.item()
            
            train_tqdm.set_postfix(train_loss=loss.item())
            
            if i % 10 == 9:  # Print every 10 mini-batches
                wandb.log({"running_loss_train": running_loss / 10}, step=(epoch+1) * len(train_dl))
                running_loss = 0.0

        model.eval()
        total_val_loss = 0
        val_tqdm = tqdm(val_dl, desc=f"Epoch {epoch + 1}/{args.epochs} Validation")
        for i, (anchor, positive, negative) in enumerate(val_tqdm):
            anchor, positive, negative = anchor.float(), positive.float(), negative.float() # shape (batch_size, 3, 1024, 2048)
            
            with torch.autocast(device_type=DEVICE, dtype=DTYPE):
                anchor_embed = model(anchor)
                positive_embed = model(positive)
                negative_embed = model(negative)
            
                # Compute Triplet Loss
                loss = triplet_loss_fn(anchor_embed, positive_embed, negative_embed)
                
            total_val_loss += loss.item()
            
            val_tqdm.set_postfix(val_loss=loss.item())
                
        wandb.log({"avg_val_loss": total_val_loss / len(val_dl)}, step=(epoch+1) * len(train_dl)) # to get it on the same axis

        torch.save({
            'model_state_dict': model.state_dict()
        }, f"epoch_{epoch + 1}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNNs/Transformers for Python code generation")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=100, help="Number of batches between logging training status to Wandb")
    parser.add_argument("--continue_from", type=str, default=None, help="Path to checkpoint file to resume training from")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    args = parser.parse_args()
    main(args)