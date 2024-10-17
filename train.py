import argparse
from itertools import cycle
import numpy as np
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

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
    
    train_ds = FmaDataset(metadata_folder="fma_metadata", root_dir="fma_processed", split="train", skip_sanity_check=args.skip_sanity_check)
    val_ds = FmaDataset(metadata_folder="fma_metadata", root_dir="fma_processed", split="val", skip_sanity_check=args.skip_sanity_check)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    TOTAL_STEPS = len(train_dl) * args.epochs
    VAL_INTERVAL = len(train_dl) // 10  # i.e. how often per epoch to validate with a portion of the validation set
    VAL_STEPS = len(val_dl) // 10  # i.e. the size of that portion
    
    train_dl = cycle(train_dl)  # infinite iterator
    val_dl = cycle(val_dl)
    
    model = Song2Vec().to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=False)  # fused speeds up training
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
        
    triplet_loss_fn = nn.TripletMarginLoss(margin=0.2, p=2)

    model = torch.compile(model, backend="aot_eager")
    model.train()

    train_loss, val_loss = float("inf"), float("inf")
    step_tqdm = tqdm(range(TOTAL_STEPS), desc="Training...")
    for step in step_tqdm:
        step_tqdm.set_description(f"Training...")
        anchor, positive, negative = next(train_dl)
        anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
        anchor, positive, negative = anchor.permute(0, 3, 1, 2), positive.permute(0, 3, 1, 2), negative.permute(0, 3, 1, 2)
        
        with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=DEVICE=="cuda"):
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
        
        if step % VAL_INTERVAL == 0 and step != 0:
            step_tqdm.set_description(f"Validating...")
            model.eval()
            total_val_loss = 0
            total_positive_cosine_distance = 0
            total_negative_cosine_distance = 0
            total_positive_2norm_distance = 0
            total_negative_2norm_distance = 0
            
            embeddings = []
            
            for _ in range(VAL_STEPS):
                anchor, positive, negative = next(val_dl)
                anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
                anchor, positive, negative = anchor.permute(0, 3, 1, 2), positive.permute(0, 3, 1, 2), negative.permute(0, 3, 1, 2)
                
                with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=DEVICE=="cuda"):
                    anchor_embed = model(anchor)
                    positive_embed = model(positive)
                    negative_embed = model(negative)
                
                    # Save anchor embeddings
                    embeddings.extend(anchor_embed.detach().cpu().tolist())
                    embeddings.extend(positive_embed.detach().cpu().tolist())
                    embeddings.extend(negative_embed.detach().cpu().tolist())
                
                    # Compute Triplet Loss
                    loss = triplet_loss_fn(anchor_embed, positive_embed, negative_embed)
                    
                    positive_cosine_distance = F.cosine_similarity(anchor_embed, positive_embed)
                    negative_cosine_distance = F.cosine_similarity(anchor_embed, negative_embed)
                    
                    positive_2norm_distance = F.pairwise_distance(anchor_embed, positive_embed)
                    negative_2norm_distance = F.pairwise_distance(anchor_embed, negative_embed)
                    
                    total_positive_cosine_distance += positive_cosine_distance.mean().item()
                    total_negative_cosine_distance += negative_cosine_distance.mean().item()
                    total_positive_2norm_distance += positive_2norm_distance.mean().item()
                    total_negative_2norm_distance += negative_2norm_distance.mean().item()
                    
                val_loss = loss.item()
                step_tqdm.set_postfix(train_loss=train_loss, val_loss=val_loss)
                total_val_loss += val_loss
            
            # Perform t-SNE on anchor embeddings
            all_embeddings = np.concatenate(embeddings)
            tsne = TSNE(n_components=3, random_state=42)
            embeddings_3d = tsne.fit_transform(all_embeddings)
            
            # Create a wandb.Table with the 3D embeddings
            columns = ["x", "y", "z"]
            data = [[x, y, z] for x, y, z in embeddings_3d]
            table = wandb.Table(data=data, columns=columns)
            
            # Log the table to wandb
            wandb.log({
                "anchor_embeddings_3d": wandb.plot_3d_scatter(table, "x", "y", "z", title="Anchor Embeddings (t-SNE 3D)"),
                "avg_val_loss": total_val_loss / VAL_STEPS,
                "avg_positive_cosine_distance": total_positive_cosine_distance / VAL_STEPS,
                "avg_negative_cosine_distance": total_negative_cosine_distance / VAL_STEPS,
                "avg_positive_2norm_distance": total_positive_2norm_distance / VAL_STEPS,
                "avg_negative_2norm_distance": total_negative_2norm_distance / VAL_STEPS,
            }, step=step)

    torch.save({
        'model_state_dict': model.state_dict()
    }, f"epoch_{args.epochs}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for song embeddings")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_interval", type=int, default=8, help="How many times per training epoch to process a correspondingly large validation portion")
    parser.add_argument("--skip_sanity_check", action="store_true")
    args = parser.parse_args()
    main(args)
