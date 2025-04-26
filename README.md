# Tuberculosis X-Ray Classification and Generation Project

This project implements a deep learning-based tuberculosis detection and synthetic image generation system using chest X-ray images. It combines a high-accuracy classifier with a Deep Convolutional Generative Adversarial Network (DCGAN) to both diagnose tuberculosis from X-rays and generate synthetic training images.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Training](#training)
  - [Training the Classifier](#training-the-classifier)
  - [Training the DCGAN](#training-the-dcgan)
- [Classification](#classification)
- [Classification Results](#classification-results)
- [Visualization (GRAD-CAM)](#visualization-grad-cam)
- [Generated Images](#generated-images)
- [Model Validation](#model-validation)
  - [Classifier Validation](#classifier-validation)
  - [DCGAN Validation](#dcgan-validation)
- [Project Outcomes](#project-outcomes)
- [Future Work](#future-work)

## Project Overview

This project addresses two key challenges in medical imaging:
1. **Tuberculosis Detection:** Developing a high-accuracy deep learning classifier to identify tuberculosis in chest X-rays
2. **Synthetic Data Generation:** Creating realistic X-ray images using DCGAN to supplement limited training data

The system achieves over 99% accuracy in tuberculosis detection while providing explainable AI through GRAD-CAM visualizations. The DCGAN component demonstrates the potential for generating synthetic medical images to address data scarcity in medical AI research.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tuberculosis-xray-project.git
cd proj_2
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

Required dependencies include:
- torch>=1.9.0
- torchvision>=0.10.0
- numpy>=1.19.5
- pandas>=1.3.0
- matplotlib>=3.4.2
- scikit-learn>=0.24.2
- pillow>=8.2.0
- opencv-python>=4.5.2
- tqdm>=4.61.1
- pytorch-grad-cam>=1.0.0

## Project Structure

- `src/`: Source code for model training and evaluation
  - `train_classifier.py`: Classifier training script
  - `train_dcgan.py`: DCGAN training script
  - `classify_images.py`: Image classification script
  - `validate_model.py`: Model validation script
  - `grad_cam.py`, `improved_grad_cam.py`: Visualization tools
  - `data_utils.py`: Data loading and processing utilities
  - `validate_dcgan.py`: DCGAN evaluation script
- `models/`: Model architecture definitions
  - `classifier.py`: ResNet and Simple CNN architecture implementations
  - `dcgan.py`: DCGAN generator and discriminator architectures
- `dataset/`: TB Chest Radiography Database
  - Contains Normal and Tuberculosis X-ray images
- `results/`: Evaluation results and visualizations
  - `validation/`: Classifier performance metrics
  - `gradcam/`: GRAD-CAM visualizations
  - `dcgan_validation/`: DCGAN evaluation results
- `output_20250423_211607/`: Latest trained models and outputs
  - `classifier/models/`: Saved classifier model weights
  - `dcgan/models/`: Saved DCGAN model weights
  - `dcgan/samples/`: Generated X-ray samples
  - `dcgan/generated/`: Additional generated images

## Dataset

The project uses the "TB Chest Radiography Database" containing:
- **Normal cases**: X-ray images of healthy patients
- **Tuberculosis cases**: X-ray images showing tuberculosis manifestations

Dataset characteristics:
- High-resolution grayscale X-ray images
- Balanced distribution of normal and tuberculosis cases
- Images are pre-processed for training (resized, normalized)

## Training

### Training the Classifier

Train the ResNet-based classifier:

```bash
python src/train_classifier.py --data_dir dataset/TB_Chest_Radiography_Database --model_type resnet --batch_size 32 --epochs 50 --learning_rate 0.0001 --output_dir output/classifier
```

Train the Simple CNN classifier:

```bash
python src/train_classifier.py --data_dir dataset/TB_Chest_Radiography_Database --model_type simple --batch_size 32 --epochs 50 --learning_rate 0.0001 --output_dir output/classifier
```

Parameters:
- `--data_dir`: Path to dataset
- `--model_type`: Model architecture (`resnet` or `simple`)
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--output_dir`: Directory to save models and results

### Training the DCGAN

Train the DCGAN on normal chest X-rays:

```bash
python src/train_dcgan.py --data_dir dataset/TB_Chest_Radiography_Database/Normal --class_name Normal --batch_size 64 --epochs 200 --output_dir output/dcgan
```

Train the DCGAN on tuberculosis chest X-rays:

```bash
python src/train_dcgan.py --data_dir dataset/TB_Chest_Radiography_Database/Tuberculosis --class_name Tuberculosis --batch_size 64 --epochs 200 --output_dir output/dcgan
```

Parameters:
- `--data_dir`: Path to class directory
- `--class_name`: Class name ('Normal' or 'Tuberculosis')
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--output_dir`: Directory to save models and generated samples

## Classification

Classify a single image:

```bash
python src/classify_images.py --model_path output_20250423_211607/classifier/models/best_classifier_model.pth --input_path dataset/TB_Chest_Radiography_Database/Tuberculosis/Tuberculosis-1.png --model_type resnet
```

Classify a directory of images:

```bash
python src/classify_images.py --model_path output_20250423_211607/classifier/models/best_classifier_model.pth --input_path dataset/TB_Chest_Radiography_Database/Tuberculosis --model_type resnet --batch_mode --output_csv results/classification_results.csv
```

Parameters:
- `--model_path`: Path to saved model weights
- `--input_path`: Path to image or directory
- `--model_type`: Model architecture (`resnet` or `simple`)
- `--batch_mode`: Enable batch processing of directory
- `--output_csv`: Path to save classification results

## Classification Results

The ResNet-based classifier achieved excellent performance on the test dataset:
- Accuracy: 99.48%
- Precision: 98.02%
- Recall: 98.86%
- F1 Score: 98.44%
- AUC: 0.9998
- PR-AUC: 0.9989

Detailed results including confusion matrices and ROC curves can be found in the `results/validation/` directory:
- `confusion_matrix.png`: Visualization of true vs. predicted classes
- `roc_curve.png`: Receiver Operating Characteristic curve
- `pr_curve.png`: Precision-Recall curve
- `probability_histogram.png`: Distribution of prediction probabilities
- `metrics_bar_chart.png`: Comparative performance metrics
- `metrics_summary.txt`: Detailed numerical results

## Visualization (GRAD-CAM)

### GRAD-CAM Implementations

Original implementation (may have issues with some layers):

```bash
python src/grad_cam.py --model_path output_20250423_211607/classifier/models/best_classifier_model.pth --input dataset/TB_Chest_Radiography_Database/Normal/Normal-1.png --model_type resnet
```

## Generated Images

View generated TB X-ray samples in:
- `output_20250423_211607/dcgan/samples/`: Examples generated during training
- `output_20250423_211607/dcgan/generated/`: Additional generated images

Generate new TB X-ray images using the trained DCGAN:

```bash
python src/generate_images.py --generator_path output_20250423_211607/dcgan/models/generator_Normal_final.pth --class_name Normal --num_images 10 --output_dir results/generated_images/normal
```

```bash
python src/generate_images.py --generator_path output_20250423_211607/dcgan/models/generator_Tuberculosis_final.pth --class_name Tuberculosis --num_images 10 --output_dir results/generated_images/tuberculosis
```

Parameters:
- `--generator_path`: Path to trained generator model
- `--class_name`: Class to generate ('Normal' or 'Tuberculosis')
- `--num_images`: Number of images to generate
- `--output_dir`: Directory to save generated images

## Model Validation

### Classifier Validation

Validate the classifier performance on the test dataset:

```bash
python src/validate_model.py --model_path output_20250423_211607/classifier/models/best_classifier_model.pth --data_dir dataset/TB_Chest_Radiography_Database --output_dir results/validation --model_type resnet
```

Parameters:
- `--model_path`: Path to saved model weights
- `--data_dir`: Path to test dataset
- `--output_dir`: Directory to save validation results
- `--model_type`: Model architecture (`resnet` or `simple`)

### DCGAN Validation

Validate the DCGAN-generated images against real images using FID score and visual inspection:

```bash
python src/validate_dcgan.py --real_dir dataset/TB_Chest_Radiography_Database/Normal --generated_dir output_20250423_211607/dcgan/samples/Normal --output_dir results/dcgan_validation/normal
```

```bash
python src/validate_dcgan.py --real_dir dataset/TB_Chest_Radiography_Database/Tuberculosis --generated_dir output_20250423_211607/dcgan/samples/Tuberculosis --output_dir results/dcgan_validation/tuberculosis
```

Parameters:
- `--real_dir`: Directory containing real images
- `--generated_dir`: Directory containing generated images
- `--output_dir`: Directory to save validation results

The validation process includes:
- Fréchet Inception Distance (FID) calculation
- Visual comparison of real vs. generated images
- Statistical analysis of image features
- Classifier performance on synthetic data

## Project Outcomes

This project demonstrates:

1. **High-performance TB detection**: The ResNet-based classifier achieves over 99% accuracy in distinguishing normal X-rays from those with tuberculosis indicators.

2. **Medical AI explainability**: GRAD-CAM visualizations highlight regions of interest in X-rays that influence the model's decision, providing transparency critical for medical applications.

3. **Synthetic medical image generation**: The DCGAN successfully generates realistic chest X-ray images that can be used to augment training data for improved classifier performance.

4. **Medical dataset augmentation**: Generated images can help address class imbalance and data scarcity issues common in medical imaging datasets.

## Future Work

Potential extensions to this project:
- Implement additional GAN architectures (StyleGAN, CycleGAN)
- Explore semi-supervised learning using generated images
- Develop a web-based interface for real-time TB detection
- Extend the model to multi-class classification for different lung diseases
- Integrate the system with DICOM image support for clinical deployment
- Optimize models for mobile deployment in resource-limited settings

