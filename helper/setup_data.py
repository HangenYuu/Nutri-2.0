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

    :return: A tuple containing two transforms.
             The first transform is for training data and
             the second transform is for validation/testing data.
    :rtype: tuple
    :raises: None
    """
    train_transforms = T.Compose([T.RandomResizedCrop(224),
                                      T.RandomRotation(35),
                                      T.RandomVerticalFlip(0.27),
                                      T.RandomHorizontalFlip(0.27),
                                      T.ToTensor(),
                                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_n_test_transforms = T.Compose([T.Resize((224,224)),
                                       T.ToTensor(),
                                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    return train_transforms, valid_n_test_transforms

def download_and_extract_data():
    """
    Downloads the data from Kaggle and extracts it.

    :return: The path to the extracted data.
    :rtype: pathlib.Path
    :raises: None
    """
    path = Path('kmader/food41')
    data_path = Path('data')
    kaggle.api.dataset_download_cli(str(path))
    if not data_path.exists():
        os.mkdir(data_path)
        zipfile.ZipFile('food41.zip').extractall(data_path)
    return data_path

def setup_folder():
    """
    Downloads and extracts the data, and sets up the folder structure
    for training and validation/testing.

    :return: The path to the extracted data.
    :rtype: pathlib.Path
    :raises: None
    """
    data_path = download_and_extract_data()
    if os.path.exists('data/train'):
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

def get_data_loaders(batch_size=64):
    """
    Returns the data loaders for training, validation, and testing.

    :param batch_size: The batch size for the data loaders.
    :type batch_size: int
    :return: A tuple containing three data loaders.
             The first data loader is for training data,
             the second data loader is for validation data,
             and the third data loader is for testing data. 
             Each data loader is an instance of the `torch.utils.data.DataLoader` class.
    :rtype: tuple
    :raises: None
    """
    data_path = setup_folder()
    train_transforms, valid_n_test_transforms = get_data_transforms()

    train_data = datasets.ImageFolder(str(data_path/'train'), transform=train_transforms)
    valid_data = datasets.ImageFolder(str(data_path/'valid'), transform=valid_n_test_transforms)
    test_data = datasets.ImageFolder(str(data_path/'test'), transform=valid_n_test_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader, test_data

def get_metadata(dataset):
    """
    Returns the metadata for the dataset.

    :param dataset: The dataset.
    :type dataset: torch.utils.data.Dataset
    :return: A tuple containing three items.
             The first item is a list of class names.
             The second item is a dictionary mapping class names to indices.
             The third item is a dictionary mapping indices to class names.
    :rtype: tuple
    :raises: None
    """
    classess, class_to_idx = dataset.classes, dataset.class_to_idx
    idx_to_class = {value: key for key, value in class_to_idx.items()}
    return classess, class_to_idx, idx_to_class
