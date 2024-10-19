import math
import argparse
from pathlib import Path

import wandb
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

torch.random.manual_seed(1337)

from baseline_model import Song2Vec
from fma_dataset import FmaDataset

def infinite_loader(data_loader):
    while True:
        for batch in data_loader:
            yield batch

CHECKPOINT_PATH = Path("checkpoints")
CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE=="cuda" else torch.float16

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"Training on device: {DEVICE}")
    
    train_ds = FmaDataset(metadata_folder="fma_metadata", root_dir="fma_processed", split="train", skip_sanity_check=args.skip_sanity_check)
    val_ds = FmaDataset(metadata_folder="fma_metadata", root_dir="fma_processed", split="val", skip_sanity_check=args.skip_sanity_check)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    TOTAL_STEPS = len(train_dl) * args.epochs
    VAL_INTERVAL = len(train_dl) // 10  # i.e. how often per epoch to validate with a portion of the validation set
    VAL_STEPS = len(val_dl) // 10  # i.e. the size of that portion
    WARMUP_STEPS = int(TOTAL_STEPS * 0.03)
    CHECKPOINT_INTERVAL = TOTAL_STEPS // 10

    MEAN = torch.tensor([-18.2629, 0.6244, 0.1782], device=DEVICE).view(1, 1, 1, 3)
    STD = torch.tensor([17.5452, 0.2233, 0.2252], device=DEVICE).view(1, 1, 1, 3)

    def normalize(audio):
        return (audio - MEAN) / STD

    def unnormalize(normalized_audio):
        return normalized_audio * STD + MEAN

    def get_lr(step:int)->float:
        if step < WARMUP_STEPS:  # 1) linear warmup for WARMUP_STEPS steps
            return args.max_lr * (step + 1) / WARMUP_STEPS
        if step > TOTAL_STEPS:  # 2) if it > TOTAL_STEPS, return min learning rate
            return args.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return args.min_lr + coeff * (args.max_lr - args.min_lr)
    
    def get_curriculum_ratio(step: int) -> tuple[float, float]:
        if step < WARMUP_STEPS:
            return 0.5, 0.5  # Start with 50% reconstruction, 50% triplet loss
        if step > TOTAL_STEPS:
            return 0.1, 0.9  # End with 10% reconstruction, 90% triplet loss
        # Linear interpolation between start and end ratios
        progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
        reconstruction_ratio = 0.5 - (0.4 * progress)  # 0.5 to 0.1
        triplet_ratio = 0.5 + (0.4 * progress)  # 0.5 to 0.9
        return reconstruction_ratio, triplet_ratio

    train_dl = infinite_loader(train_dl)  # infinite iterator
    val_dl = infinite_loader(val_dl)
    
    model = Song2Vec().to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=args.max_lr, fused=False)  # fused speeds up training
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.9)
    
    print(f"training model with {sum([p.numel() for p in model.parameters() if p.requires_grad])/1e6:.2f}M parameters")

    wandb.init(
        name=args.run_name,
        project="Song2Vec",
        config={
            "learning_rate": args.max_lr,
            "epochs": args.epochs,
            "n_training_examples": len(train_ds),
            "n_validation_examples": len(val_ds),
            "parameter_count": sum([p.numel() for p in model.parameters() if p.requires_grad]),
            **vars(args)
        },
    )
    
    reconstruction_loss_fn = nn.MSELoss()
    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

    model = torch.compile(model, backend="aot_eager")
    model.train()

    train_loss, val_loss = float("inf"), float("inf")
    step_tqdm = tqdm(range(TOTAL_STEPS), desc="Training...")
    for step in step_tqdm:
        step_tqdm.set_description(f"Training...")
        anchor, positive, negative = next(train_dl)
        anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
        anchor, positive, negative = normalize(anchor), normalize(positive), normalize(negative)

        lr = get_lr(step)
        reconstruction_ratio, triplet_ratio = get_curriculum_ratio(step)
        for param_group in optim.param_groups: param_group["lr"] = lr

        with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=DEVICE=="cuda"):
            anchor_out, anchor_embed = model(anchor)
            positive_embed, _ = model.encode(positive)  # no need to decode positive / negative
            negative_embed, _ = model.encode(negative)

            anchor_out = unnormalize(anchor_out)
        
            reconstruction_loss = reconstruction_loss_fn(anchor_out, anchor) * 0.1  # to account for the inherent summing over the time axis
            triplet_loss = triplet_loss_fn(anchor_embed, positive_embed, negative_embed)
            loss = reconstruction_loss * reconstruction_ratio + triplet_loss * triplet_ratio

        positive_cosine_similarity = F.cosine_similarity(anchor_embed, positive_embed)
        negative_cosine_similarity = F.cosine_similarity(anchor_embed, negative_embed)
        
        # Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        train_loss = loss.item()
        wandb.log({
            "lr": lr,
            "reconstruction_ratio": reconstruction_ratio,
            "loss/total_loss": train_loss,
            "loss/triplet": triplet_loss.item(),
            "loss/reconstruction": reconstruction_loss.item(),
            "cosine_similarity/positive": positive_cosine_similarity.mean().item(),
            "cosine_similarity/negative": negative_cosine_similarity.mean().item()
        }, step=step)
        step_tqdm.set_postfix(train_loss=train_loss, val_loss=val_loss)
        
        if step % VAL_INTERVAL == 0 and step != 0:
            with torch.no_grad():
                step_tqdm.set_description(f"Validating...")
                model.eval()
                total_val_loss = 0
                total_triplet_loss = 0
                total_reconstruction_loss = 0
                total_positive_cosine_similarity = 0
                total_negative_cosine_similarity = 0
                
                for _ in range(VAL_STEPS):
                    anchor, positive, negative = next(val_dl)
                    anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
                    
                    with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=DEVICE=="cuda"):
                        anchor_out, anchor_embed = model(anchor)
                        positive_embed, _ = model.encode(positive)
                        negative_embed, _ = model.encode(negative)

                        anchor_out = unnormalize(anchor_out)

                        reconstruction_loss = reconstruction_loss_fn(anchor_out, anchor) * 0.1  # to account for the inherent summing over the time axis
                        triplet_loss = triplet_loss_fn(anchor_embed, positive_embed, negative_embed)
                        loss = reconstruction_loss * reconstruction_ratio + triplet_loss * triplet_ratio
                        
                    positive_cosine_similarity = F.cosine_similarity(anchor_embed, positive_embed)
                    negative_cosine_similarity = F.cosine_similarity(anchor_embed, negative_embed)

                    val_loss = loss.item()
                    total_val_loss += val_loss
                    total_triplet_loss += triplet_loss.item()
                    total_reconstruction_loss += reconstruction_loss.item()
                    total_positive_cosine_similarity += positive_cosine_similarity.mean().item()
                    total_negative_cosine_similarity += negative_cosine_similarity.mean().item()
                    
                    step_tqdm.set_postfix(train_loss=train_loss, val_loss=val_loss)

                wandb.log({
                    "loss/avg_val_total": total_val_loss / VAL_STEPS,
                    "loss/avg_val_triplet": total_triplet_loss / VAL_STEPS,
                    "loss/avg_val_reconstruction": total_reconstruction_loss / VAL_STEPS,
                    "cosine_similarity/avg_val_positive": total_positive_cosine_similarity / VAL_STEPS,
                    "cosine_similarity/avg_val_negative": total_negative_cosine_similarity / VAL_STEPS,
                }, step=step)
            model.train()

        if step % CHECKPOINT_INTERVAL == 0 and step != 0:
            run_path = CHECKPOINT_PATH / args.run_name
            run_path.mkdir(parents=True, exist_ok=True)
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict()
            }, run_path / f"step_{step}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for song embeddings")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_lr", type=float, default=3e-4, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Minimum learning rate")
    parser.add_argument("--val_interval", type=int, default=8, help="How many times per training epoch to process a correspondingly large validation portion")
    parser.add_argument("--skip_sanity_check", action="store_true")
    args = parser.parse_args()
    main(args)
