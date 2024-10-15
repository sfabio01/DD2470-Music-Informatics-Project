import argparse
from itertools import cycle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb
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
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # TODO: compute our own
    ])

    train_ds = FmaDataset(metadata_folder="fma_metadata", root_dir="fma_processed", split="train", transform = transform)
    val_ds = FmaDataset(metadata_folder="fma_metadata", root_dir="fma_processed", split="val", transform = transform)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    TOTAL_STEPS = len(train_dl) * args.epochs
    VAL_INTERVAL = len(train_dl) // 10
    VAL_STEPS = len(val_dl) // VAL_INTERVAL
    
    train_dl = cycle(train_dl)  # infinite iterator
    val_dl = cycle(val_dl)
    
    model = Song2Vec().to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=DEVICE=="cuda")  # fused speeds up training
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

    # model = torch.compile(model, backend="aot_eager")
    model.train()

    train_loss, val_loss = float("inf"), float("inf")
    step_tqdm = tqdm(range(TOTAL_STEPS), desc="Training...")
    for step in step_tqdm:
        step_tqdm.set_description(f"Training...")
        anchor, positive, negative = next(train_dl)
        anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
        
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
        
        train_loss = loss.item()
        wandb.log({"train_loss": train_loss}, step=step)
        step_tqdm.set_postfix(train_loss=train_loss, val_loss=val_loss)
        
        if step % VAL_STEPS == 0 and step != 0:
            step_tqdm.set_description(f"Validating...")
            model.eval()
            total_val_loss = 0
            
            for _ in range(VAL_STEPS):
                anchor, positive, negative = next(val_dl)
                anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
                
                with torch.autocast(device_type=DEVICE, dtype=DTYPE):
                    anchor_embed = model(anchor)
                    positive_embed = model(positive)
                    negative_embed = model(negative)
                
                    # Compute Triplet Loss
                    loss = triplet_loss_fn(anchor_embed, positive_embed, negative_embed)
                
                val_loss = loss.item()
                step_tqdm.set_postfix(train_loss=train_loss, val_loss=val_loss)
                total_val_loss += val_loss
            
            wandb.log({"val_loss": total_val_loss / VAL_STEPS}, step=step)

    torch.save({
        'model_state_dict': model.state_dict()
    }, f"epoch_{args.epochs}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for song embeddings")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    main(args)