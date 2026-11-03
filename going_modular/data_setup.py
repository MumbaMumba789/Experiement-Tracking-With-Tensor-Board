#library imports
import os
import torch.nn as nn
import torch
import requests
import zipfile
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#data accessing
# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# If the image folder doesn't exist, download it and prepare it... 
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)
    
    # Download pizza, steak, sushi data
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...") 
        zip_ref.extractall(image_path)


#data prep
def create_dataloaders(
    train_dir,
    test_dir,
    transform,
    batch_size,
    num_workers=0
):

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=transform
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_dataloader, test_dataloader, train_dataset.classes