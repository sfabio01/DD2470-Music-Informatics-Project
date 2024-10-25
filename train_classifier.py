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
import sklearn

from classifier import GenreClassifier
from classification_dataset import MyDataset

torch.random.manual_seed(1337)

from model import Song2Vec

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
    
    train_ds = MyDataset(metadata_folder="fma_metadata", root_dir="fma_processed", split="train", skip_sanity_check=args.skip_sanity_check)
    val_ds = MyDataset(metadata_folder="fma_metadata", root_dir="fma_processed", split="val", skip_sanity_check=args.skip_sanity_check)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    TOTAL_STEPS = len(train_dl) * args.epochs
    VAL_INTERVAL = len(train_dl) // 10  # i.e. how often per epoch to validate with a portion of the validation set
    VAL_STEPS = len(val_dl) // 10  # i.e. the size of that portion
    WARMUP_STEPS = int(TOTAL_STEPS * 0.03)
    CHECKPOINT_INTERVAL = TOTAL_STEPS // 10


    # def get_lr(step:int)->float:
    #     if step < WARMUP_STEPS:  # 1) linear warmup for WARMUP_STEPS steps
    #         return args.max_lr * (step + 1) / WARMUP_STEPS
    #     if step > TOTAL_STEPS:  # 2) if it > TOTAL_STEPS, return min learning rate
    #         return args.min_lr
    #     # 3) in between, use cosine decay down to min learning rate
    #     decay_ratio = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    #     assert 0 <= decay_ratio <= 1
    #     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    #     return args.min_lr + coeff * (args.max_lr - args.min_lr)
    

    train_dl = infinite_loader(train_dl)  # infinite iterator
    val_dl = infinite_loader(val_dl)
    
    song2vec = Song2Vec().to(DEVICE)
    # load weights from file
    if args.model_path is not None:
        state_dict = torch.load(args.model_path, map_location=DEVICE)["model_state_dict"]
        wo_orig_mod = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        song2vec.load_state_dict(wo_orig_mod)
    
    model = GenreClassifier(song2vec, num_genres=8).to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.9)
    
    print(f"training model with {sum([p.numel() for p in model.parameters() if p.requires_grad])/1e6:.2f}M parameters")

    wandb.init(
        name=args.run_name,
        project="Song2Vec",
        config={
            "learning_rate": args.max_lr,
            "weight_decay": 1e-3,
            "epochs": args.epochs,
            "n_training_examples": len(train_ds),
            "n_validation_examples": len(val_ds),
            "parameter_count": sum([p.numel() for p in model.parameters() if p.requires_grad]),
            **vars(args)
        },
    )
    
    ce_loss = nn.CrossEntropyLoss()

    model = torch.compile(model, backend="aot_eager")
    model.train()

    train_loss, val_loss = float("inf"), float("inf")
    step_tqdm = tqdm(range(TOTAL_STEPS), desc="Training...")
    for step in step_tqdm:
        step_tqdm.set_description(f"Training...")
        images, labels = next(train_dl)
        images = images.to(DEVICE)


        with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=DEVICE=="cuda"):
            predictions = model(images)
        
            loss = ce_loss(predictions, labels)

        
        # Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        train_loss = loss.item()
        wandb.log({
            "loss/total_loss": train_loss
        }, step=step)
        step_tqdm.set_postfix(train_loss=train_loss, val_loss=val_loss)
        
        if step % VAL_INTERVAL == 0 and step != 0:
            with torch.no_grad():
                step_tqdm.set_description(f"Validating...")
                model.eval()
                total_val_loss = 0
                all_predictions = []
                all_labels = []
            
                for _ in range(VAL_STEPS):
                    images, labels = next(val_dl)
                    images = images.to(DEVICE)
                    
                    with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=DEVICE=="cuda"):
                        predictions = model(images)

                    loss = ce_loss(predictions, labels)
                    
                    val_loss = loss.item()
                    total_val_loss += val_loss

                    all_predictions.extend(predictions.argmax(dim=1).cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    step_tqdm.set_postfix(train_loss=train_loss, val_loss=val_loss)


                avg_val_loss = total_val_loss / VAL_STEPS
                total_accuracy = sklearn.metrics.accuracy_score(all_labels, all_predictions)
                per_class_accuracy = sklearn.metrics.precision_score(all_labels, all_predictions, average=None)
                total_f1_score = sklearn.metrics.f1_score(all_labels, all_predictions, average='weighted')

                wandb.log({
                    "loss/avg_val_total": avg_val_loss,
                    "metrics/total_accuracy": total_accuracy,
                    "metrics/per_class_accuracy": per_class_accuracy,
                    "metrics/total_f1_score": total_f1_score,
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
    parser.add_argument("--model_path", type=str, default=None, help="Path to Song2Vec checkpoint")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_lr", type=float, default=3e-4, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Minimum learning rate")
    parser.add_argument("--val_interval", type=int, default=8, help="How many times per training epoch to process a correspondingly large validation portion")
    parser.add_argument("--skip_sanity_check", action="store_true")
    args = parser.parse_args()
    main(args)
