import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sklearn.metrics as metrics
from tqdm import tqdm
from PIL import Image

# Import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classifier import XRayClassifier, SimpleXRayClassifier
from src.data_utils import XRayDataset


def load_model(model_path, model_type='resnet', device=None):
    """
    Load a trained classifier model.
    
    Args:
        model_path: Path to the saved model weights (.pth file)
        model_type: Type of model architecture ('simple' or 'resnet')
        device: Device to run inference on (cuda or cpu)
        
    Returns:
        The loaded model and device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model architecture
    if model_type.lower() == 'resnet':
        model = XRayClassifier(num_classes=2, pretrained=False).to(device)
    else:  # 'simple'
        model = SimpleXRayClassifier(num_classes=2).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    
    print(f"Model loaded from {model_path}")
    return model, device


def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on the given data loader.
    
    Args:
        model: The trained classifier model
        data_loader: DataLoader for the dataset to evaluate
        device: Device to run inference on (cuda or cpu)
        
    Returns:
        Dictionary containing evaluation metrics and arrays needed for further analysis
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    # No gradient calculation needed for evaluation
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # Convert to numpy arrays
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])  # Probability of class 1 (TB)
    
    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = metrics.accuracy_score(all_targets, all_preds) * 100
    precision = metrics.precision_score(all_targets, all_preds, zero_division=0) * 100
    recall = metrics.recall_score(all_targets, all_preds, zero_division=0) * 100
    f1 = metrics.f1_score(all_targets, all_preds, zero_division=0) * 100
    conf_matrix = metrics.confusion_matrix(all_targets, all_preds)
    
    # Calculate ROC curve points and AUC
    fpr, tpr, thresholds = metrics.roc_curve(all_targets, all_probs)
    auc = metrics.auc(fpr, tpr)
    
    # Calculate precision-recall curve
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(all_targets, all_probs)
    pr_auc = metrics.auc(recall_curve, precision_curve)
    
    # Calculate class-wise metrics
    classification_report = metrics.classification_report(all_targets, all_preds, 
                                                         target_names=['Normal', 'Tuberculosis'])
    
    # Return metrics
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'fpr': fpr,
        'tpr': tpr,
        'auc': auc,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
        'pr_auc': pr_auc,
        'all_probs': all_probs,
        'all_targets': all_targets,
        'all_preds': all_preds,
        'classification_report': classification_report
    }


def plot_confusion_matrix(cm, output_path, class_names=['Normal', 'Tuberculosis']):
    """
    Plot and save a confusion matrix.
    
    Args:
        cm: Confusion matrix array
        output_path: Path to save the figure
        class_names: Names of the classes
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")


def plot_roc_curve(fpr, tpr, auc, output_path):
    """
    Plot and save a ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc: Area under curve value
        output_path: Path to save the figure
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_path)
    print(f"ROC curve saved to {output_path}")


def plot_precision_recall_curve(recall, precision, pr_auc, output_path):
    """
    Plot and save a precision-recall curve.
    
    Args:
        recall: Recall values
        precision: Precision values
        pr_auc: Area under precision-recall curve
        output_path: Path to save the figure
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.axhline(y=0.5, color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(output_path)
    print(f"Precision-Recall curve saved to {output_path}")


def plot_probability_histogram(probs, targets, output_path, bins=20):
    """
    Plot and save a histogram of prediction probabilities by class.
    
    Args:
        probs: Predicted probabilities for positive class
        targets: True labels
        output_path: Path to save the figure
        bins: Number of histogram bins
    """
    plt.figure(figsize=(10, 6))
    
    # Separate probabilities by actual class
    normal_probs = probs[targets == 0]
    tb_probs = probs[targets == 1]
    
    # Plot the histograms
    plt.hist(normal_probs, bins=bins, alpha=0.5, color='green', label='Normal')
    plt.hist(tb_probs, bins=bins, alpha=0.5, color='red', label='Tuberculosis')
    
    plt.xlabel('Probability of Tuberculosis Class')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Probabilities by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    print(f"Probability histogram saved to {output_path}")


def plot_metrics_bar_chart(metrics_dict, output_path):
    """
    Create a simple bar chart of key metrics.
    
    Args:
        metrics_dict: Dictionary containing accuracy, precision, recall, and f1
        output_path: Path to save the figure
    """
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_values = [metrics_dict['accuracy'], metrics_dict['precision'], 
                      metrics_dict['recall'], metrics_dict['f1']]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'red', 'purple'])
    
    # Add the values on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, 105)  # Leave some room for annotations
    plt.ylabel('Percentage (%)')
    plt.title('Classification Metrics')
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(output_path)
    print(f"Metrics bar chart saved to {output_path}")


def run_validation(data_dir, model_path, output_dir, model_type='resnet', batch_size=32, image_size=128):
    """
    Run validation on a dataset and generate all metrics and visualizations.
    
    Args:
        data_dir: Directory containing validation data with 'Normal' and 'Tuberculosis' subfolders
        model_path: Path to the saved model weights
        output_dir: Directory to save results
        model_type: Type of model architecture ('simple' or 'resnet')
        batch_size: Batch size for evaluation
        image_size: Size to resize images to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, device = load_model(model_path, model_type, device)
    
    # Create data transformation
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create dataset and data loader
    dataset = XRayDataset(data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Evaluating model on {len(dataset)} images")
    
    # Evaluate model
    metrics_dict = evaluate_model(model, data_loader, device)
    
    # Save metrics as text file
    with open(os.path.join(output_dir, 'metrics_summary.txt'), 'w') as f:
        f.write(f"Model Type: {model_type.upper()}\n")
        f.write(f"Accuracy: {metrics_dict['accuracy']:.2f}%\n")
        f.write(f"Precision: {metrics_dict['precision']:.2f}%\n")
        f.write(f"Recall: {metrics_dict['recall']:.2f}%\n")
        f.write(f"F1 Score: {metrics_dict['f1']:.2f}%\n")
        f.write(f"AUC: {metrics_dict['auc']:.4f}\n")
        f.write(f"PR-AUC: {metrics_dict['pr_auc']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(metrics_dict['confusion_matrix'], separator=', '))
        f.write("\n\nDetailed Classification Report:\n")
        f.write(metrics_dict['classification_report'])
    
    # Generate all plots
    plot_confusion_matrix(metrics_dict['confusion_matrix'], 
                         os.path.join(output_dir, 'confusion_matrix.png'))
    
    plot_roc_curve(metrics_dict['fpr'], metrics_dict['tpr'], metrics_dict['auc'], 
                  os.path.join(output_dir, 'roc_curve.png'))
    
    plot_precision_recall_curve(metrics_dict['recall_curve'], metrics_dict['precision_curve'], 
                               metrics_dict['pr_auc'], os.path.join(output_dir, 'pr_curve.png'))
    
    plot_probability_histogram(metrics_dict['all_probs'], metrics_dict['all_targets'],
                              os.path.join(output_dir, 'probability_histogram.png'))
    
    plot_metrics_bar_chart(metrics_dict, os.path.join(output_dir, 'metrics_bar_chart.png'))
    
    print(f"All validation metrics and visualizations saved to {output_dir}")
    
    return metrics_dict


def main():
    parser = argparse.ArgumentParser(description='Validate TB X-ray Classification Model')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory containing test data with Normal and Tuberculosis subfolders')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to trained model weights')
    parser.add_argument('--output_dir', type=str, default='results/validation',
                        help='Directory to save validation results')
    parser.add_argument('--model_type', type=str, choices=['simple', 'resnet'], default='resnet',
                        help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Size to resize images to')
    
    args = parser.parse_args()
    
    # Run validation
    run_validation(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        model_type=args.model_type,
        batch_size=args.batch_size,
        image_size=args.image_size
    )


if __name__ == '__main__':
    main()