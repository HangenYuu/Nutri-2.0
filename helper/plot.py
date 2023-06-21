"""
Contains functions for data visualization and plotting
"""
import torch
import matplotlib.pyplot as plt

def imshow(image, ax=None, title=None):
    """
    imshow for IMAGENET-style transformed images in
    torch.Tensor format.

    Args:
        image (torch.Tensor): Image to be plotted
        ax (matplotlib.axes.Axes): Axes object to plot the image on
        title (str): Title for the image
    
    Returns:
        ax (matplotlib.axes.Axes): Axes object with the image plotted
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.permute((1, 2, 0))
    
    # Undo preprocessing
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    image = (std * image + mean)
    
    
    ax.imshow(image)
    if title is not None:
        ax.set_title(title)
    
    return ax

def image_grid(images, labels, idx_to_class, rows=2, cols=4, figsize=(20, 10)):
    """
    Plots a grid of images and their labels.

    Args:
        images (torch.Tensor): Images to be plotted
        labels (torch.Tensor): Labels for the images
        idx_to_class (dict): Dictionary mapping indices to class names
        rows (int): Number of rows in the grid
        cols (int): Number of columns in the grid
        figsize (tuple): Size of the figure
    
    Returns:
        None
    """
    _, axes = plt.subplots(rows, cols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        ax = imshow(images[i], ax=ax, title=idx_to_class[labels[i].item()])
    plt.tight_layout()
    plt.show()

def plot_training(train_losses, valid_losses, train_accuracies, valid_accuracies):
    """
    Plots the training and validation losses and accuracies.
    
    Args:
        train_losses (list): List of training losses
        valid_losses (list): List of validation losses
        train_accuracies (list): List of training accuracies
        valid_accuracies (list): List of validation accuracies
    
    Returns:
        None
    """
    num_epochs = len(train_losses)
    epochs = range(1, num_epochs + 1)

    # Plot subplots for loss and accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot loss subplot
    ax1.plot(epochs, train_losses, label='Training Loss')
    ax1.plot(epochs, valid_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    # Plot accuracy subplot
    ax2.plot(epochs, train_accuracies, label='Training Accuracy')
    ax2.plot(epochs, valid_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()

    # Show the plot
    plt.show()
