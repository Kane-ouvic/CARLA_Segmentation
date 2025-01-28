# Project Overview

This project is focused on image segmentation using various deep learning models. The primary goal is to train, validate, and infer segmentation models on a given dataset, utilizing different architectures and techniques to improve performance.

## Project Structure

- **main.py**: Entry point of the program. It sets up the data loaders, model, loss function, and optimizer, and initiates the training and inference processes.
- **train.py**: Contains functions for training and validating models, as well as saving model weights and plotting training progress.
- **dataset.py**: Handles data loading and augmentation. It defines custom dataset classes for segmentation tasks.
- **utils.py**: Defines various loss functions used during training.
- **Unet.py, UnetPlusPlus.py, UnetPlusPlusImproved.py**: Define different UNet-based model architectures.
- **EfficientNet.py**: Implements an improved UNet++ architecture using EfficientNet as the encoder.
- **inference.py**: Script for running inference on test images and saving the results.
- **postprocess.py**: Applies post-processing techniques like CRF to refine segmentation results.
- **count.py**: Calculates pixel distribution in the predicted masks.
- **filter.py**: Applies denoising filters to images.
- **vote.py**: Implements a voting mechanism to combine predictions from multiple models.

## Installation

1. Clone the repository
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train a model, modify the `main.py` script to select the desired model architecture and hyperparameters.
