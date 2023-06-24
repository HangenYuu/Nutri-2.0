"""
Contains functions to download and setup the data
"""
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
import kaggle
import zipfile
import shutil
import os
import json
from pathlib import Path

def get_data_transforms():
    """
    Returns the data transforms for training and validation/testing.

    Args:
        None
    
    Returns:
        train_transforms (torchvision.transforms.Compose): Transforms for training data
        valid_n_test_transforms (torchvision.transforms.Compose): Transforms for validation and testing data
    """
    train_transforms = T.Compose([T.RandomResizedCrop(224),
                                      T.RandomRotation(35),
                                      T.RandomVerticalFlip(0.27),
                                      T.RandomHorizontalFlip(0.27),
                                      T.ToTensor(),
                                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = T.Compose([T.Resize((224,224)),
                                       T.ToTensor(),
                                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    return train_transforms, test_transforms

def download_and_extract_data():
    """
    Downloads the data from Kaggle and extracts it.

    Args:
        None
    
    Returns:
        data_path (pathlib.Path): Path to the extracted data
    """
    data_path = Path('data')
    if not data_path.exists():
        os.mkdir(data_path)
        path = Path('kmader/food41')
        kaggle.api.dataset_download_cli(str(path))
        zipfile.ZipFile('food41.zip').extractall(data_path)
    return data_path

def setup_folder():
    """
    Downloads and extracts the data, and sets up the folder structure
    for training and validation/testing.

    Args:
        None

    Returns:
        data_path (pathlib.Path): Path to the extracted data
    """
    data_path = download_and_extract_data()
    if os.path.exists(data_path/'train'):
        return data_path
    
    with open(data_path/'meta/meta/train.json', 'r') as fp:
        train_dict = json.load(fp)
    with open(data_path/'meta/meta/test.json', 'r') as fp:
        test_dict = json.load(fp)

    new_folders = ['train', 'test']
    for folder in new_folders:
        if not os.path.exists(data_path/folder):
            os.mkdir(data_path/folder)
        if folder == 'train':
            if not os.path.exists(data_path/'valid'):
                os.mkdir(data_path/'valid')
            for key, value in train_dict.items():
                train_value, valid_value = train_test_split(value, train_size=0.75)
                train_set, valid_set = set(train_value), set(valid_value)
                if not os.path.exists(data_path/folder/key):
                    os.mkdir(data_path/folder/key)
                if not os.path.exists(data_path/'valid'/key):
                    os.mkdir(data_path/'valid'/key)
                for image in os.listdir(data_path/'images'/key):
                    image_path = key + '/' + image
                    image_id = image_path.split('.')[0]
                    if image_id in train_set:
                        shutil.move(data_path/'images'/image_path, data_path/folder/image_path)
                    if image_id in valid_set:
                        shutil.copy(data_path/'images'/image_path, data_path/'valid'/image_path)
        else:
            for key, value in test_dict.items():
                test_set = set(value)
                if not os.path.exists(data_path/folder/key):
                    os.mkdir(data_path/folder/key)
                for image in os.listdir(data_path/'images'/key):
                    image_path = key + '/' + image
                    image_id = image_path.split('.')[0]
                    if image_id in test_set:
                        shutil.move(data_path/'images'/image_path, data_path/folder/image_path)
    shutil.rmtree(data_path/'images')
    return data_path

def get_data_loaders(train_transforms, test_transforms, batch_size=64, num_workers=0):
    """
    Returns the data loaders for training, validation, and testing.

    Args:
        train_transforms (torchvision.transforms.Compose): Transforms for training data
        test_transforms (torchvision.transforms.Compose): Transforms for validation and testing data
        batch_size (int): Batch size for the data loaders
        num_workers (int): Number of workers for the data loaders
        
    Returns:
        train_loader (torch.utils.data.DataLoader): Data loader for training data
        valid_loader (torch.utils.data.DataLoader): Data loader for validation data
        test_loader (torch.utils.data.DataLoader): Data loader for testing data
        test_data (torchvision.datasets.ImageFolder): Testing data
    """
    data_path = setup_folder()

    train_data = datasets.ImageFolder(str(data_path/'train'), transform=train_transforms)
    valid_data = datasets.ImageFolder(str(data_path/'valid'), transform=test_transforms)
    test_data = datasets.ImageFolder(str(data_path/'test'), transform=test_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader, test_loader, test_data

def get_metadata(dataset):
    """
    Returns the metadata for the dataset.

    Args:
        dataset (torchvision.datasets.ImageFolder): Dataset for which metadata is required
    
    Returns:
        classes (list): List of class names
        class_to_idx (dict): Dictionary mapping class names to indices
        idx_to_class (dict): Dictionary mapping indices to class names
    """
    classes, class_to_idx = dataset.classes, dataset.class_to_idx
    idx_to_class = {value: key for key, value in class_to_idx.items()}
    return classes, class_to_idx, idx_to_class
