import torch
import matplotlib.pyplot as plt

def imshow(image, ax=None, title=None):
    """
    imshow for IMAGENET-style transformed images in
    torch.Tensor format.

    :param image: The image to be plotted.
    :type image: torch.Tensor
    :param ax: The axis to plot the image on.
    :type ax: matplotlib.axes.Axes
    :param title: The title of the plot.
    :type title: str
    :return: The axis with the plotted image.
    :rtype: matplotlib.axes.Axes
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

    :param images: The images to be plotted.
    :type images: torch.Tensor
    :param labels: The labels of the images.
    :type labels: torch.Tensor
    :param class_to_idx: A dictionary mapping class names to indices.
    :type class_to_idx: dict
    :param idx_to_class: A dictionary mapping indices to class names.
    :type idx_to_class: dict
    :param rows: The number of rows in the grid.
    :type rows: int
    :param cols: The number of columns in the grid.
    :type cols: int
    :param figsize: The size of the figure.
    :type figsize: tuple
    :return: None
    :rtype: None
    """
    _, axes = plt.subplots(rows, cols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        ax = imshow(images[i], ax=ax, title=idx_to_class[labels[i].item()])
    plt.tight_layout()
    plt.show()
