import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from PIL import Image
from tqdm import tqdm

# Import custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classifier import XRayClassifier, SimpleXRayClassifier
from src.data_utils import XRayDataset, get_data_loaders

class CombinedDataset(Dataset):
    """Dataset that combines real and generated X-ray images."""
    
    def __init__(self, real_dataset, generated_dir, class_name, transform=None):
        """
        Args:
            real_dataset: Original XRayDataset
            generated_dir: Directory containing generated images
            class_name: Class of generated images ('Normal' or 'Tuberculosis')
            transform: Optional transform to be applied on images
        """
        self.real_dataset = real_dataset
        self.transform = transform
        
        # Determine the class label for the generated images
        self.class_label = 0 if class_name == 'Normal' else 1
        
        # Load paths to generated images
        self.generated_paths = []
        gen_class_dir = os.path.join(generated_dir, class_name)
        if os.path.exists(gen_class_dir):
            for file_name in os.listdir(gen_class_dir):
                if file_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.generated_paths.append(os.path.join(gen_class_dir, file_name))
        
        print(f"Loaded {len(self.generated_paths)} generated {class_name} images")
    
    def __len__(self):
        return len(self.real_dataset) + len(self.generated_paths)
    
    def __getitem__(self, idx):
        if idx < len(self.real_dataset):
            # Return a real image
            return self.real_dataset[idx]
        else:
            # Return a generated image
            gen_idx = idx - len(self.real_dataset)
            img_path = self.generated_paths[gen_idx]
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, self.class_label

def train_classifier(data_dir, generated_dir, output_dir, image_size=128, batch_size=32, 
                    epochs=20, lr=0.001, use_generated=True, model_type='simple',
                    train_test_split=0.2, val_split=0.1, seed=42, device=None):
    """
    Train a classifier to distinguish between TB and non-TB X-ray images.
    
    Args:
        data_dir: Directory containing TB_Chest_Radiography_Database dataset
        generated_dir: Directory containing generated images
        output_dir: Directory to save models and results
        image_size: Size to resize images to
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
        use_generated: Whether to use generated images in training
        model_type: Type of classifier model ('simple' or 'resnet')
        train_test_split: Proportion of data to use for testing
        val_split: Proportion of training data to use for validation
        seed: Random seed for reproducibility
        device: Device to use (cuda or cpu)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create datasets
    dataset_path = os.path.join(data_dir, 'TB_Chest_Radiography_Database')
    xray_dataset = XRayDataset(dataset_path, transform=transform)
    
    # Split into train, validation, and test sets
    dataset_size = len(xray_dataset)
    test_size = int(train_test_split * dataset_size)
    train_val_size = dataset_size - test_size
    val_size = int(val_split * train_val_size)
    train_size = train_val_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        xray_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Combine with generated images if specified
    if use_generated and os.path.exists(generated_dir):
        print("Adding generated images to training set...")
        normal_gen_dataset = CombinedDataset(train_dataset, generated_dir, 'Normal', transform)
        tb_gen_dataset = CombinedDataset(normal_gen_dataset, generated_dir, 'Tuberculosis', transform)
        train_dataset = tb_gen_dataset
    
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
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    if model_type.lower() == 'resnet':
        model = XRayClassifier(num_classes=2, pretrained=True).to(device)
    else:  # 'simple'
        model = SimpleXRayClassifier(num_classes=2, channels=3, image_size=image_size).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    # Lists to track progress
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(model_dir, 'best_classifier_model.pth')
    
    print("Starting classifier training...")
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for i, (images, labels) in train_progress:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_progress.set_postfix({
                'loss': train_loss / (i + 1),
                'accuracy': 100. * train_correct / train_total
            })
        
        train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_progress = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
        with torch.no_grad():
            for i, (images, labels) in val_progress:
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Track statistics
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                val_progress.set_postfix({
                    'loss': val_loss / (i + 1),
                    'accuracy': 100. * val_correct / val_total
                })
        
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
    
    # Save the final model
    final_model_path = os.path.join(model_dir, 'final_classifier_model.pth')
    torch.save(model.state_dict(), final_model_path)
    
    # Plot training and validation losses
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'))
    
    # Evaluate the model on the test set
    print("\nEvaluating model on test set...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='weighted') * 100
    recall = recall_score(all_labels, all_preds, average='weighted') * 100
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Precision: {precision:.2f}%")
    print(f"Test Recall: {recall:.2f}%")
    print(f"Test F1 Score: {f1:.2f}%")
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Tuberculosis'], 
                yticklabels=['Normal', 'Tuberculosis'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    
    # Save metrics to file
    with open(os.path.join(results_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f"Model type: {model_type}\n")
        f.write(f"Generated images used: {use_generated}\n")
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")
        f.write(f"Test Precision: {precision:.2f}%\n")
        f.write(f"Test Recall: {recall:.2f}%\n")
        f.write(f"Test F1 Score: {f1:.2f}%\n")
    
    print(f"Training complete. Results saved to {results_dir}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train classifier for TB vs non-TB X-ray images')
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Directory containing the dataset')
    parser.add_argument('--generated_dir', type=str, default='./data/generated/generated', help='Directory containing generated images')
    parser.add_argument('--output_dir', type=str, default='./results/classifier', help='Directory to save output')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--image_size', type=int, default=128, help='Image size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model_type', type=str, choices=['simple', 'resnet'], default='simple', help='Model architecture')
    parser.add_argument('--use_generated', action='store_true', help='Use generated images for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    train_classifier(
        data_dir=args.data_dir,
        generated_dir=args.generated_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        epochs=args.epochs,
        lr=args.lr,
        use_generated=args.use_generated,
        model_type=args.model_type,
        seed=args.seed
    )