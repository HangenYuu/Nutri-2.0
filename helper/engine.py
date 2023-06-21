"""
Contains functions for training and evaluating the model.
"""
from typing import Dict, List, Tuple
import gc
import torch
import torchmetrics
from tqdm.auto import tqdm

def cuda_collect():
    """
    Collects the garbage and empties the cache on GPU.
    """
    gc.collect()
    torch.cuda.empty_cache()

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler.LRScheduler,
               accuracy_fn: torchmetrics.Accuracy,
               device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    train_loss, train_acc = 0, 0
    model.train()
    # Model is expected to be in GPU already
    for _, (images, labels) in tqdm(enumerate(data_loader),
                                        total=len(data_loader),
                                        desc='Training model:'):
        images, labels= images.to(device), labels.to(device)

        # 1. Forward pass
        preds = model(images)

        # 2. Calculate loss
        loss = loss_fn(preds, labels)
        train_loss += loss
        train_acc += accuracy_fn(preds.argmax(dim=1), labels)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # 6. Garbage collection on GPU RAM
        if device == torch.device('cuda'):
            cuda_collect()
    scheduler.step()
    # Exponential learning rate scheduler reduces learning rate too fast
    # A better option is torch.optim.lr_scheduler.OneCycleLR from the paper
    # of https://arxiv.org/pdf/1708.07120.pdf on MNIST dataset
    # I changed from exponential learning rate to onecycle learning scheduler
    # as performance became slow after 70 epochs.

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}")
    return train_loss.cpu(), train_acc.cpu()

def test_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn: torchmetrics.Accuracy,
               device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    test_loss, test_acc = 0, 0
    model.eval()
    # Turn on inference context manager
    with torch.inference_mode():
        for images, labels in tqdm(data_loader,
                                    total=len(data_loader),
                                    desc='Making predictions:'):

            images, labels= images.to(device), labels.to(device)

            # 1. Forward pass
            preds = model(images)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(preds, labels)
            test_acc += accuracy_fn(preds.argmax(dim=1), labels)

            if device == torch.device('cuda'):
                cuda_collect()

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}")
    return test_loss.cpu(), test_acc.cpu()

def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          valid_loader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.LRScheduler,
          accuracy_fn: torchmetrics.Accuracy,
          device: torch.device,
          epochs: int,
          threshold: List[float]) -> Dict[str, List[torch.Tensor]]:
    """
    Trains the model and evaluates it on the validation set.
    
    Args:
        model (torch.nn.Module): Model to be trained
        train_loader (torch.utils.data.DataLoader): Training data loader
        valid_loader (torch.utils.data.DataLoader): Validation data loader
        loss_fn (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler
        accuracy_fn (torchmetrics.Accuracy): Accuracy function
        device (torch.device): Device to run the training on
        epochs (int): Number of epochs to train the model for
        threshold (float): Threshold for early stopping
    
    Returns:
        Dictionary containing training and validation losses and accuracies for each epoch.
        In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 
        For example if training for epochs=2: 
                    {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]} 
    """
    results = {"train_losses": [], "train_accuracies": [],
               "valid_losses": [], "valid_accuracies": []}
    
    tolerance = 0
    threshold = torch.Tensor(threshold)
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, scheduler, accuracy_fn, device)
        valid_loss, valid_acc = test_step(model, valid_loader, loss_fn, accuracy_fn, device)
        print(
            f"Epoch {epoch + 1} of {epochs}"
            f"\n-------------------------------"
            f"\nTrain loss: {train_loss:.5f} | Train accuracy: {train_acc:.4f}"
            f"\nValid loss: {valid_loss:.5f} | Valid accuracy: {valid_acc:.4f}"
        )

        results["train_losses"].append(train_loss.detach())
        results["train_accuracies"].append(train_acc.cpu())
        results["valid_losses"].append(valid_loss.detach())
        results["valid_accuracies"].append(valid_acc.cpu())
        if len(results["valid_losses"]) > 1 and results["valid_losses"][-2] - results["valid_losses"][-1] < threshold:
            tolerance += 1
            if tolerance > 2:
                break
    return results
