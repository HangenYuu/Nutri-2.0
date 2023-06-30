"""
Contains functions for training and evaluating the model.
"""
from typing import Dict, List, Tuple
import gc
import torch
import torchmetrics
from tqdm import tqdm
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter

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
               accuracy_fn: torchmetrics.Metric,
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
               accuracy_fn: torchmetrics.Metric,
               device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    test_loss, test_acc = 0, 0
    model.eval()
    # Turn on inference context manager
    with torch.no_grad():
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

def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str = '') -> torch.utils.tensorboard.writer.SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          valid_loader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.LRScheduler,
          accuracy_fn: torchmetrics.Metric,
          device: torch.device,
          epochs: int,
          writer: torch.utils.tensorboard.SummaryWriter = None,
          threshold: List[float] = [0]) -> Dict[str, List[torch.Tensor]]:
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
        writer (torch.utils.tensorboard.SummaryWriter): SummaryWriter instance. Defaults to None.
        threshold (List[float], optional): Threshold for early stopping. Defaults to [0].
                                           Value is put inside a list for easy conversion to torch.Tensor.
    
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
    if writer:
        writer.add_graph(model=model, 
                         input_to_model=torch.randn(32, 3, 224, 224).to(device))

    for epoch in range(epochs):
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

        # Tensorboard logging
        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(main_tag="Loss", 
                               tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": valid_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag="Accuracy", 
                               tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": valid_acc}, 
                               global_step=epoch)

            # Close the writer
            writer.close()

        if len(results["valid_losses"]) > 1 and results["valid_losses"][-2] - results["valid_losses"][-1] < threshold:
            tolerance += 1
            if tolerance > 2:
                break
    return results
