import torch
import os
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.optim import AdamW
from grokfast import gradfilter_ema
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from model import device
from dataset import get_stock_data_by_symbol_ibkr, create_labels_local_min_max

CKPT_DIR = "checkpoints"

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)

def save_model(model: nn.Module, epoch: int, loss: float, name: str = None):
    """Saves model checkpoint as a state dict.

    Args:
        model (nn.Module): Model.
        epoch (int): Training epoch.
        loss (float): Current loss.
        name (str, optional): Name added to the filename of the checkpoint. Defaults to None.
    """
    name = "" if name is None else name.lower() + "-"
    checkpoint_dir = Path(__file__).resolve().parent / CKPT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_name = model._get_name()
    state_dict = model.state_dict()
    with open(checkpoint_dir / f"{model_name}_{name}{epoch}_{loss:.3f}.pt") as file:
        torch.save(state_dict, file)

def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.modules.loss._Loss) -> list[float]:
    """Evaluates the model based on a validation dataset.

    Args:
        model (nn.Module): Model to be validated.
        dataloader (DataLoader): Validation data.
        criterion (nn.modules.loss._Loss): Loss function.

    Returns:
        list[float]: List of batch validation losses.
    """
    losses = []
    model = model.eval().to(device)
    for i, (x_batch, y_batch) in tqdm(enumerate(dataloader), desc="Validating...", total=len(dataloader)):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        losses.append(loss.item())
    logging.debug(f"Completed validation with an average validation loss of {sum(losses)/len(losses):.3f}.")
    return losses

def train_grokfast(model: nn.Module, dataloader: DataLoader, criterion: nn.modules.loss._Loss, n_epochs: int = 20, plot: bool = False):
    """Trains model using grokfast for faster generalization.

    Args:
        model (nn.Module): Model to be trained.
        dataloader (DataLoader): Training data in batches.
        criterion (nn.modules.loss._Loss): Loss function.
        n_epochs (int, optional): Number of epochs to train the model for. Defaults to 20.
        plot (bool, optional): Whether to plot the training loss progression after completing training. Defaults to False.

    Returns:
        nn.Module: Trained model.
    """
    model = model.to(device)
    optimizer = AdamW(
        model.parameters(), 
        lr=1e-3,
        )
    scheduler = StepLR(optimizer, step_size=n_epochs//2, gamma=0.5)
    losses = []
    grads = None
    # training loop
    for epoch in range(n_epochs):
        epoch_losses = []
        for i, (x_batch, y_batch) in (pbar := tqdm(enumerate(dataloader), desc=f"Epoch {epoch} | Loss [n/a]", total=len(dataloader))):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            # gradient clipping (not really necessary as we use LayerNorm)
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 10)
            # grokfast (faster generalization)
            grads = gradfilter_ema(model, grads=grads, alpha=0.98, lamb=2.0)
            optimizer.step()
            epoch_losses.append(loss.item())
            # update progress bar loss every 10 steps
            if i % 10 == 0:
                pbar.set_description(f"Epoch {epoch} | Loss {loss.item():.3f}")
        logging.debug(f"Completed epoch {epoch} with an average training loss of {sum(epoch_losses)/len(epoch_losses):.3f}.")
        losses.extend(epoch_losses)
        scheduler.step()
    # plot training losses
    if plot:
        plt.plot(losses, list(range(len(losses))))
        plt.title("Training loss")
        plt.xlabel("Loss")
        plt.ylabel("Step")
        plt.show()
    return model


if __name__ == "__main__":
    model = None
    criterion = nn.MSELoss()
    dataloader = None
    train_grokfast(model, dataloader, criterion, n_epochs=20)
