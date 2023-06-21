"""
Contains utility functions for training and saving model
"""
import torch
from pathlib import Path

def save_model(model, save_dir, model_name, epoch, optimizer, loss):
    """
    Saves model checkpoint

    Args:
        model (torch.nn.Module): Model to be saved
        save_dir (str): Directory to save model
        model_name (str): Name of model
        epoch (int): Epoch number
        optimizer (torch.optim.Optimizer): Optimizer
        loss (float): Loss value

    Returns:
        None
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir / f'{model_name}_epoch{epoch}_loss{loss}.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print(f'[INFO] Model saved to {save_path}')
