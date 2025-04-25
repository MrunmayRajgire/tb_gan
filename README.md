# Tuberculosis X-Ray Classification and Generation Project

This project implements a tuberculosis detection system using chest X-ray images, with deep learning models for classification and image generation.

## Table of Contents
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

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd proj_2
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

- `src/`: Source code for model training and evaluation
- `models/`: Model architecture definitions
- `dataset/`: TB Chest Radiography Database
- `results/`: Evaluation results and visualizations
- `output_*/`: Trained models and outputs

## Dataset

The project uses the "TB Chest Radiography Database" containing X-rays of:
- Normal cases
- Tuberculosis cases

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

Detailed results including confusion matrices and ROC curves can be found in the `results/validation/` directory.

## Visualization (GRAD-CAM)

### Basic GRAD-CAM (Recommended)

Generate GRAD-CAM for a single image:

```bash
python src/basic_grad_cam.py --input dataset/TB_Chest_Radiography_Database/Tuberculosis/Tuberculosis-162.png --model output_20250423_211607/classifier/models/best_classifier_model.pth --output results/basic_gradcam/tb_visualization.png --model_type resnet
```

Process a batch of images with GRAD-CAM:

```bash
python src/basic_grad_cam.py --input dataset/TB_Chest_Radiography_Database/Tuberculosis --model output_20250423_211607/classifier/models/best_classifier_model.pth --output results/basic_gradcam/tb_batch --model_type resnet --batch
```

Parameters:
- `--input`: Path to input image or directory
- `--model`: Path to trained model weights
- `--output`: Path to save visualization(s)
- `--model_type`: Model type ('resnet' or 'simple')
- `--batch`: Process images in batch mode

### Alternative GRAD-CAM Implementations

Original implementation (may have issues with some layers):

```bash
python src/grad_cam.py --model_path output_20250423_211607/classifier/models/best_classifier_model.pth --input dataset/TB_Chest_Radiography_Database/Normal/Normal-1.png --model_type resnet
```

Improved implementation (alternative approach):

```bash
python src/improved_grad_cam.py --input dataset/TB_Chest_Radiography_Database/Tuberculosis/Tuberculosis-162.png --model output_20250423_211607/classifier/models/best_classifier_model.pth --output results/improved_gradcam/tb_visualization.png --model_type resnet
```

## Generated Images

View generated TB X-ray samples in:
- `output_20250423_211607/dcgan/samples/`

Generate new TB X-ray images using trained DCGAN:

```bash
# Coming soon - use DCGAN to generate new images
```

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

Validate the DCGAN-generated images against real images using FID score:

```bash
python src/validate_dcgan.py --class_name Normal --epoch 50 --data_dir dataset/TB_Chest_Radiography_Database --output_dir results/dcgan_validation/normal
```

```bash
python src/validate_dcgan.py --class_name Tuberculosis --epoch 50 --data_dir dataset/TB_Chest_Radiography_Database --output_dir results/dcgan_validation/tuberculosis
```

Parameters:
- `--class_name`: Class to validate ('Normal' or 'Tuberculosis')
- `--epoch`: Epoch number of the trained model to use
- `--data_dir`: Path to dataset directory
- `--output_dir`: Directory to save validation results

For the Normal class, the DCGAN achieved an FID score of 578.1152, indicating moderate similarity between generated and real images. Lower FID scores indicate better quality and similarity.

---

For any issues or questions, please open an issue in the repository.