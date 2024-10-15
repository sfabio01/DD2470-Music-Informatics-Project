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
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.9)
    
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
        total_train_loss = 0

        for i, batch in enumerate(train_tqdm):
            batch = batch.to(DEVICE)
            x = batch[..., :-1]
            y = batch[..., 1:]
            
            with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE):
                y_hat = model(x)
                if isinstance(y_hat, tuple): y_hat = y_hat[0]
                loss = criterion(y_hat.reshape(-1, config.vocab_size), y.reshape(-1))

            optim.zero_grad()
           

            train_loss = loss.detach().cpu().numpy()
            total_train_loss += train_loss
            train_tqdm.set_postfix({"loss": f"{train_loss:.3f}"})

            if i % args.log_interval == 0:
                wandb.log({"train_loss": train_loss}, step=epoch * len(train_dl) + i, commit=True)

        scheduler.step()
        wandb.log({"avg_train_loss": total_train_loss / len(train_dl)}, step=(epoch+1) * len(train_dl)) # to get it on the same axis

        model.eval()
        total_val_loss = 0.0
        total_val_perplex = 0.0
        with torch.no_grad():
            val_tqdm = tqdm(val_dl, desc=f"Epoch {epoch + 1}/{start_epoch + args.epochs} Validation")
            for val_batch in val_tqdm:
                val_batch = val_batch.to(DEVICE)
                x_val = val_batch[..., :-1]
                y_val = val_batch[..., 1:]

                with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE):
                    y_hat = model(x_val)
                    if isinstance(y_hat, tuple): y_hat = y_hat[0]
                    loss = criterion(y_hat.reshape(-1, config.vocab_size), y_val.reshape(-1))
                    
                val_loss = loss.detach().cpu().numpy()
                total_val_loss += val_loss
                val_tqdm.set_postfix({"val_loss": f"{val_loss:.3f}"})
                
                total_val_perplex += np.exp(val_loss)
                
        wandb.log({"avg_val_loss": total_val_loss / len(val_dl)}, step=(epoch+1) * len(train_dl)) # to get it on the same axis
        wandb.log({"avg_val_perplexity": total_val_perplex / len(val_dl)}, step=(epoch+1) * len(train_dl))
        
        batch = next(iter(val_dl)).to(DEVICE)
        B, L = batch.shape
        context = int(0.25 * L)
        pred_length = 2 * context
        x = batch[:, :context]
        y = batch[:, context:pred_length]
        y_hat = model.generate(B, max_len=L, starting_tokens=x, nucleus_threshold=0.5)
        y_hat = [seq[context:pred_length] for seq in y_hat]
        avg_bleu_score = bleu_score(y.tolist(), y_hat, n_gram=4)
                    
        gen = model.generate(B, max_len=200, nucleus_threshold=0.5)
        programs = [tokenizer.detokenize(gen_seq) for gen_seq in gen]
        avg_syntax_error_score = syntax_error_score(programs)
        
        wandb.log({"avg_bleu4_score": avg_bleu_score}, step=(epoch+1) * len(train_dl))
        wandb.log({"avg_syntax_error_score": avg_syntax_error_score}, step=(epoch+1) * len(train_dl))
        wandb.log({"generated_text": wandb.Html(tokenizer.color_text_html(programs[0]))}, step=(epoch+1) * len(train_dl))

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': total_train_loss / len(train_dl),
            'wandb_id': wandb.run.id,
            'config_name': args.config
        }, model_path / f"epoch_{epoch + 1}.pt")


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