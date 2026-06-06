# FOTS (Fast Oriented Text Spotting) - Unified Text Detection & Recognition

## Project Overview
This repository contains a high-performance implementation of the **FOTS (Fast Oriented Text Spotting)** architecture, an end-to-end trainable framework for simultaneous text detection and recognition in natural scenes. By sharing features between the detection and recognition branches, FOTS achieves significant performance gains in both speed and accuracy compared to traditional multi-stage OCR pipelines.

## What is this Project?
The project provides a comprehensive research and development environment for advanced text spotting:
- **Unified End-to-End Pipeline:** Unlike separate detection and recognition models, this unified architecture allows for joint optimization of both tasks.
- **Oriented Text Support:** Specifically designed to detect and read text at any angle using **RBOX (Rotated Bounding Box)** geometry.
- **Production-Ready Inference:** Includes optimized inference logic and supports **TensorFlow Lite (TFLite)** for high-efficiency deployment.

## Neural Network Architecture Details
The architecture is composed of four primary components:
1.  **Shared Convolutional Features (Backbone):** A deep CNN (typically ResNet50 or similar) that extracts high-level semantic features from the input image.
2.  **Feature Merging Branch:** A custom layer (`feature_merging_branch`) that implements a Feature Pyramid Network (FPN) style approach. It uses `UpSampling2D` (bilinear interpolation) and `Concatenate` to merge low-level spatial features with high-level semantic features, ensuring accurate detection of text at multiple scales.
3.  **Text Detection Branch:**
    - Predicts a **Score Map** (probability of a pixel being text).
    - Predicts a **Geometry Map (RBOX)**: 4 channels for distances to top, right, bottom, and left boundaries ($d_1, d_2, d_3, d_4$) and 1 channel for the rotation angle ($\theta$).
4.  **ROIRotate & Text Recognition Branch:**
    - Uses **ROIRotate** to transform oriented feature regions into fixed-size feature maps.
    - An RNN-based decoder (often LSTM/GRU) predicts the final text sequence.

## Loss Functions (Technical Implementation)
The model is trained using a multi-task loss function:
- **Detection Loss ($L_{det}$):** A weighted combination of:
    - **Dice Loss:** For the score map classification, handling the class imbalance between text and background pixels.
    - **IOU Loss:** Calculated on the predicted bounding boxes to ensure precise localization.
    - **Angle Loss:** $1 - \cos(\hat{\theta} - \theta)$, encouraging the model to predict the correct orientation.
- **Recognition Loss ($L_{rec}$):** Uses **CTC (Connectionist Temporal Classification) Loss**, allowing the model to be trained on unaligned sequences.

## Tech Stack
- **Deep Learning:** TensorFlow 2.x, Keras
- **Geometry Operations:** Shapely (for polygon and IOU calculations)
- **Computer Vision:** OpenCV (`cv2`), PIL
- **Data Engineering:** NumPy, Pandas, Scikit-learn
- **Deployment:** TensorFlow Lite (TFLite)
- **Serialization:** Pickle (for tokenizers), CSV

## Key Features
- **Custom Keras Layers:** Implementation of `feature_merging_branch` and specialized loss classes.
- **Complex Geometry Reconstruction:** `restore_rectangle_rbox` logic to convert model outputs back into readable coordinates.
- **Interactive Training:** Comprehensive Jupyter Notebooks (`FOTS_data_prepare.ipynb`, `pipeline.ipynb`) documenting the entire data science lifecycle.
- **Streamlit Integration:** `develop.py` provides a real-time web interface for testing the spotting pipeline.

## File Descriptions
- `ml_models.py`: The core engine containing architecture definitions, custom losses (`detection_loss`, `ctc_loss`), and the `inferencePipeline_lite`.
- `FOTS_data_prepare.ipynb`: Detailed guide on preparing oriented text datasets (e.g., ICDAR) for training.
- `pipeline.ipynb`: Demonstrates the full spotting workflow from raw image to extracted text.
- `develop.py`: Streamlit application for interactive text detection.
