# fots_development

## Project Overview
Development repository for a Fast Oriented Text Spotting (FOTS) model or similar text detection and recognition pipeline.

## What is this Project?
This project aims to detect and recognize text within images using advanced computer vision techniques and unified deep learning models.

## How it was done
The project uses Jupyter notebooks and Python scripts to prepare data and build machine learning pipelines. `FOTS_data_prepare.ipynb` handles the complex preprocessing of image and bounding box data required for text spotting. The deep learning models and pipeline architecture are defined in `ml_models.py` and `pipeline.ipynb`.

## Why it was done
To implement or experiment with end-to-end text spotting architectures (like FOTS) which combine text detection and text recognition into a single, efficient neural network.

## Tech Stack
- Python
- Computer Vision libraries (OpenCV, PIL)
- Deep Learning frameworks (TensorFlow, PyTorch, or Keras)
- Jupyter Notebook

## Key Features
- Specialized data preparation for oriented text detection.
- Unified machine learning pipeline for spotting text in images.
- Scripted model definitions for reproducibility.

## File Structure
- `FOTS_data_prepare.ipynb`: Notebook for preprocessing text detection datasets.
- `pipeline.ipynb`: Main notebook orchestrating the training or evaluation pipeline.
- `ml_models.py`: Python module containing the neural network architectures.
- `develop.py`: Development or utility script.

## Local Setup (if applicable)
1. Clone the repository.
2. Install standard CV and ML libraries: `pip install opencv-python tensorflow numpy pandas jupyter`.
3. Open the notebooks to review the data preparation and pipeline execution.