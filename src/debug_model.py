import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classifier import XRayClassifier, SimpleXRayClassifier

def print_model_layers(model, model_type):
    """Print all layers in the model with their paths"""
    print(f"\n=== {model_type.upper()} MODEL ARCHITECTURE ===")
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d, 
                             torch.nn.ReLU, torch.nn.MaxPool2d, torch.nn.Sequential)):
            print(f"Layer: {name}, Type: {type(module).__name__}")

def main():
    # Create both model types
    resnet_model = XRayClassifier(num_classes=2, pretrained=False)
    simple_model = SimpleXRayClassifier(num_classes=2)
    
    # Print model architectures
    print_model_layers(resnet_model, "ResNet")
    print_model_layers(simple_model, "Simple")
    
    # For ResNet, also print the layer4's structure specifically
    if hasattr(resnet_model, 'model') and hasattr(resnet_model.model, 'layer4'):
        print("\n=== DETAILED RESNET LAYER4 STRUCTURE ===")
        for idx, bottleneck in enumerate(resnet_model.model.layer4):
            print(f"Layer4[{idx}]: {bottleneck.__class__.__name__}")
            for name, module in bottleneck.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    print(f"  - {name}: {module.__class__.__name__}, out_channels: {module.out_channels}")

if __name__ == "__main__":
    main()