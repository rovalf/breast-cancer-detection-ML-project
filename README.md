# Deep Learning for Breast Cancer Detection

An AI-assisted mammogram classification system using Convolutional Neural Networks (CNNs) with Grad-CAM explainability.  
This project was developed as part of the University of London Final Year Project, with the aim of demonstrating the feasibility of deep learning in improving breast cancer screening workflows.

---

## Overview

Breast cancer remains one of the most prevalent cancers worldwide, where early detection is critical to survival. Radiologists face increasing workload and shortages, which can lead to diagnostic delays. This project explores the use of deep learning to assist radiologists by providing automated predictions and visual explanations on mammogram images.

The system integrates:

- **Deep Learning (CNNs)** for binary classification of mammograms (benign vs. malignant).
- **EfficientNet backbones** (B0–B3) fine-tuned on the CBIS-DDSM dataset.
- **Grad-CAM visualizations** to highlight suspicious regions, supporting interpretability and clinical trust.
- **Flask web interface** that allows uploading mammograms in DICOM/PNG/JPG format, conversion to preview images, and generating predictions with corresponding heatmaps.

---

## Features

- Upload mammograms in **DICOM** format → automatic conversion to PNG preview.  
- Classify images as **benign** or **malignant** with confidence scores.  
- Generate **Grad-CAM overlays** for visual explanation of model predictions.  
- Modular design with a fallback mechanism to handle different model formats (`.keras` / `.h5`).  
- Secure login with session-based authentication.  

---

## Technical Stack

- **Python** (Flask, NumPy, TensorFlow/Keras, Pillow, OpenCV, pydicom)  
- **Deep Learning Models:** EfficientNet-B0, B2, B3 backbones  
- **Dataset:** Curated Breast Imaging Subset of DDSM (CBIS-DDSM)  
- **Explainability:** Grad-CAM (Gradient-weighted Class Activation Mapping)  

---

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/rovalf/breast-cancer-detection-ML-project.git
   cd breast-cancer-detection-ML-project
