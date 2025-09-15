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

2. Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


3. Install dependencies:

pip install -r requirements.txt


4. Run the application:

python app.py


5. Access the web interface at:
http://localhost:5000

Dataset

This project uses the Curated Breast Imaging Subset of the Digital Database for Screening Mammography (CBIS-DDSM).
The dataset contains decompressed, segmented mammogram images with corresponding pathology labels, curated to support reproducible research.

Reference:

Lee, R.S., Gimenez, F., Hoogi, A., Miyake, K.K. and Rubin, D.L. (2017) A curated mammography dataset for use in computer-aided detection and diagnosis research, Scientific Data, 4, p.170177. Link

Key References

Lee, R.S. et al. (2017) A curated mammography dataset for use in computer-aided detection and diagnosis research, Scientific Data.

Wang, L. (2024) Mammography with Deep Learning for Breast Cancer Detection, Frontiers in Oncology.

Selvaraju, R.R. et al. (2017) Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization, International Conference on Computer Vision (ICCV).

Breast Cancer Surveillance Consortium (BCSC) benchmarks: https://www.bcsc-research.org/statistics/screening-performance-benchmarks

Expected Outcomes

A trained CNN model capable of achieving competitive sensitivity/specificity against BCSC benchmarks.

A web-based prototype demonstrating both classification and interpretability features.

Evaluation against literature-reported results, with analysis of strengths, limitations, and clinical feasibility.
