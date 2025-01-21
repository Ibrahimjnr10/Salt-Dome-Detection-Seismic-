# Salt-Dome-Detection-Seismic-
Seismic Image Segmentation for Salt Domes Detection in Resource Exploration

# U-Net for Grayscale Image Segmentation

This project implements a U-Net model for semantic segmentation of grayscale images using PyTorch. The model is trained to predict binary masks for given input images, with applications in areas like seismic data interpretation or medical image analysis.

## Project Overview

The U-Net model is a convolutional neural network (CNN) architecture designed for efficient image segmentation. This implementation processes grayscale images and generates segmentation masks that can be used to highlight regions of interest in the images.

The project includes:
- Data loading and transformation
- U-Net model implementation with residual blocks
- Model training with loss and evaluation using Intersection over Union (IoU)
- Inference pipeline for generating predictions on test images

## Prerequisites

To run this project, you need to install the following Python libraries:

- `torch` (PyTorch)
- `numpy`
- `pandas`
- `matplotlib`
- `PIL` (Pillow)
- `torchvision`

You can install the required dependencies using the following command:

```bash
pip install torch torchvision numpy pandas matplotlib pillow
```

### Dataset
The dataset consists of grayscale images and corresponding binary masks. The images are resized to 96x96 pixels for model processing.

### Model Architecture
- U-Net with residual blocks for feature extraction.
- Downsampling with convolutional layers and upsampling with deconvolution layers.
- Binary cross-entropy loss (BCEWithLogitsLoss) and IoU score for evaluation.

### Training
The model is trained using:

- Adam optimizer
- Batch size: 64
- Epochs: 3
Training involves computing loss and IoU metrics for each batch.

### Inference
After training, the model generates binary masks for new images. The process involves:

- Loading a trained model.
- Preprocessing input images.
- Running the model to predict masks

### License
This project is licensed under the MIT License.
