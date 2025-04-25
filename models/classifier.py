import torch
import torch.nn as nn
import torchvision.models as models

class XRayClassifier(nn.Module):
    """
    CNN-based classifier for TB vs non-TB X-ray images.
    Uses a pre-trained ResNet model as a backbone with custom classifier head.
    """
    
    def __init__(self, num_classes=2, pretrained=True, feature_extracting=True):
        """
        Initialize the classifier.
        
        Args:
            num_classes: Number of output classes (2 for TB vs non-TB)
            pretrained: Whether to use pretrained weights for the backbone
            feature_extracting: If True, freeze the backbone and only train the classifier head
        """
        super(XRayClassifier, self).__init__()
        
        # Initialize ResNet-50 model
        self.model = models.resnet50(pretrained=pretrained)
        
        # Freeze parameters if doing feature extraction
        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False
                
        # Replace the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass of the classifier.
        
        Args:
            x: Input image tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Logits for each class
        """
        return self.model(x)

class SimpleXRayClassifier(nn.Module):
    """
    A simpler CNN-based classifier for TB vs non-TB X-ray images.
    """
    
    def __init__(self, num_classes=2, channels=3, image_size=128):
        """
        Initialize the classifier.
        
        Args:
            num_classes: Number of output classes (2 for TB vs non-TB)
            channels: Number of input channels (3 for RGB)
            image_size: Size of input images
        """
        super(SimpleXRayClassifier, self).__init__()
        
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate the size of the feature maps after the convolutional blocks
        feature_size = image_size // 16  # After 4 max-pooling layers with stride 2
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * feature_size * feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass of the classifier.
        
        Args:
            x: Input image tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Logits for each class
        """
        x = self.features(x)
        x = self.classifier(x)
        return x