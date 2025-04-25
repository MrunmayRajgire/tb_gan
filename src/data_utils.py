import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from tqdm import tqdm

class XRayDataset(Dataset):
    """Dataset class for TB and Normal X-ray images."""
    
    def __init__(self, data_dir, transform=None, class_labels=None):
        """
        Args:
            data_dir (str): Directory with TB and Normal subfolders
            transform (callable, optional): Optional transform to be applied on a sample.
            class_labels (dict, optional): Dictionary mapping folder names to class indices.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.class_labels = {'Normal': 0, 'Tuberculosis': 1} if class_labels is None else class_labels
        
        self.image_paths = []
        self.labels = []
        
        for class_name in self.class_labels.keys():
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_labels[class_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders(data_dir, batch_size=16, image_size=128, train_val_split=0.2, seed=42):
    """
    Create train and validation data loaders.
    
    Args:
        data_dir: Directory containing TB_Chest_Radiography_Database dataset
        batch_size: Batch size for training
        image_size: Size to resize images to
        train_val_split: Proportion of data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create dataset
    full_dataset = XRayDataset(data_dir, transform=transform)
    
    # Split into train and validation sets
    dataset_size = len(full_dataset)
    val_size = int(train_val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader

def save_generated_images(generated_images, save_dir, prefix="generated"):
    """
    Save generated images to disk.
    
    Args:
        generated_images: Tensor of generated images [batch_size, channels, height, width]
        save_dir: Directory to save images in
        prefix: Prefix for image filenames
    """
    os.makedirs(save_dir, exist_ok=True)
    transform = transforms.ToPILImage()
    
    for i, img_tensor in enumerate(generated_images):
        # Denormalize the image tensor from [-1, 1] to [0, 1]
        img_tensor = (img_tensor.clone() + 1) / 2.0
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        # Convert to PIL Image and save
        img = transform(img_tensor)
        img.save(os.path.join(save_dir, f"{prefix}_{i}.png"))