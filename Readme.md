# Retinal Vessel Segmentation and Automated Diabetic Risk Detection

## 👋 Welcome

Welcome to the **Retinal Vessel Segmentation and Automated Diabetic Risk Detection** repository.

We are delighted to present this comprehensive research project, which stands at the intersection of **Medical Imaging** and **Artificial Intelligence**. This repository is designed to provide researchers, developers, and medical professionals with a transparent, in-depth, and reproducible framework for automated retinal analysis.

Whether you are here to explore traditional computer vision algorithms for vessel extraction or to investigate state-of-the-art deep learning models for disease classification, we hope this documentation serves as a valuable resource in your journey.

To get started immediately, please refer to the [Implementation & Usage](#-implementation--usage) section.

---

## 🧾 Executive Summary

This project presents a comprehensive framework for the automated analysis of retinal fundus images, addressing two critical diagnostic challenges: **Retinal Vessel Segmentation** and **Diabetic Retinopathy Risk Detection**. 

By integrating traditional **Computer Vision** techniques (Signal Processing, Morphological Operations) with modern **Deep Learning** architectures (Transfer Learning with VGG16), this system provides a robust tool for enhancing vessel visibility and classifying disease risk. The project is architected with modularity in mind, separating core logic into reusable Python modules for scalability.

---

## 📌 Table of Contents
- [Key Features](#-key-features)
- [Technical Architecture](#-technical-architecture)
- [Part 1: Vessel Segmentation (Image Processing)](#part-1-vessel-segmentation-image-processing)
- [Part 2: Diabetic Risk Detection (Deep Learning)](#part-2-diabetic-risk-detection-deep-learning)
- [Implementation & Usage](#-implementation--usage)
- [Results & Performance](#-results--performance)
- [References](#-references)

---

## 🚀 Key Features

*   **Hybrid Approach**: Combines the interpretability of traditional image processing with the predictive power of Deep Learning.
*   **Advanced Preprocessing**: Implements a custom pipeline including Green Channel Extraction, CLAHE, and Non-Local Means Denoising.
*   **Robust Segmentation**: Utilizes Gabor Matched Filtering and Adaptive Thresholding for precise vessel extraction without the need for labeled training masks.
*   **Automated Classification**: Deploys a VGG16-based classifier with a custom head for high-accuracy diabetic risk detection.
*   **Modular Codebase**: Core algorithms are encapsulated in the `model/` directory for easy integration and testing.

---

## 🏗 Technical Architecture

The project is structured to separate data processing, modeling, and evaluation:

```
├── model/                  # Core Logic Modules
│   ├── preprocessing.py    # Image cropping, CLAHE, and enhancement functions
│   ├── segmentation.py     # End-to-end vessel segmentation pipeline
│   └── train_classifier.py # VGG16 training loop, augmentation, and evaluation
├── Data/                   # Raw and processed data for segmentation
├── datasets/               # Organized dataset for Classification (Normal/Disease)
├── classification.ipynb    # Initial classification experiments
├── final.ipynb             # Interactive vessel segmentation notebook
└── ml_implementation.ipynb # Interactive Deep Learning notebook
```

---

## Part 1: Vessel Segmentation (Image Processing)

The segmentation module relies on the physiological property that blood vessels appear darker than the background in the green channel of the spectrum.

### 1. Preprocessing Pipeline

#### 1.1 Green Channel Extraction
The green channel is extracted from the RGB image as it provides the highest contrast between blood vessels and the background retina. The red channel is often saturated, and the blue channel is noisy.
![RGB Channels](Images/rgb_channels.png)

#### 1.2 Contrast Enhancement (CLAHE)
**Contrast Limited Adaptive Histogram Equalization (CLAHE)** is applied to enhance local contrast. Unlike standard Histogram Equalization (HE) which operates globally, CLAHE operates on small regions (tiles) of the image.
*   **Mechanism**: The image is divided into 8x8 tiles. Histogram equalization is performed on each tile.
*   **Clip Limit (2.0)**: To prevent noise amplification in homogeneous areas, the histogram is clipped at a specific limit before equalization.
![CLAHE](Images/clahe.png)

#### 1.3 Noise Reduction
*   **Non-Local Means Denoising**: A sophisticated technique that averages pixels based on the similarity of their surrounding patches, effectively removing noise while preserving fine vessel structures.
![Non-local Means](Images/non_local_means.png)


![Enhancement Comparison](Images/enhancement_comparison.png)



### 2. Vessel Detection Methodology

#### 2.1 Matched Filtering (Gabor Filters)
We employ **Gabor Filters** to detect tubular structures (vessels). A bank of Gabor kernels with varying orientations is convolved with the image. The filter response is maximal when the kernel aligns with a vessel segment, effectively highlighting the vascular network.
![Matched Filtering](Images/matched_filtering.png)

#### 2.2 Adaptive Thresholding
To convert the enhanced grayscale image into a binary mask, we use **Adaptive Thresholding**. The threshold value is calculated dynamically for smaller regions, allowing the method to handle varying illumination conditions across the retina.
![Adaptive Threshold](Images/adaptive_threshold.png)

---

## Part 2: Diabetic Risk Detection (Deep Learning)

For the classification task (Normal vs. Diabetic), the project utilizes **Transfer Learning** to leverage features learned from large-scale datasets (ImageNet).

### 1. Data Pipeline
*   **Black Border Removal**: A custom cropping function (`crop_image_from_gray` in `model/preprocessing.py`) removes the uninformative black borders around the fundus image, centering the region of interest.
*   **Data Augmentation**: To improve model generalization, `ImageDataGenerator` applies:
    *   Rotation (+/- 20 degrees)
    *   Zoom (+/- 20%)
    *   Horizontal Flip
    *   Shift (Width/Height)

### 2. Model Architecture: VGG16
We utilize the **VGG16** architecture as a feature extractor.
*   **Base**: VGG16 (weights frozen) extracts hierarchical features (edges, textures, shapes).
*   **Custom Head**:
    *   `GlobalAveragePooling2D`: Reduces spatial dimensions.
    *   `Dense (128 units, ReLU)`: Learns specific features for diabetic retinopathy.
    *   `Dropout (0.5)`: Regularization to prevent overfitting.
    *   `Dense (Output, Softmax)`: Final classification layer.

---

## 💻 Implementation & Usage

### Prerequisites
*   Python 3.9+
*   TensorFlow 2.8+
*   OpenCV, NumPy, Matplotlib, Scikit-learn

### 1. Running Vessel Segmentation
To segment vessels from a single image using the standalone script:

```bash
cd model
python segmentation.py "path/to/your/image.jpg"
```
*   **Output**: A binary mask `vessel_mask.jpg` will be saved in the current directory.

### 2. Training the Classifier
To train the Deep Learning model on your dataset:

1.  Ensure your dataset is organized in `datasets/` (e.g., `datasets/normal/`, `datasets/diabetes/`).
2.  Run the training script:
    ```bash
    cd model
    python train_classifier.py
    ```
    *   **Process**: The script loads data, applies augmentation, trains the VGG16 model (with early stopping), and displays evaluation metrics.
    *   **Output**: The trained model is saved as `diabetic_retinopathy_model.h5`.

---

## 📊 Results & Performance

### Image Processing
*   **Enhancement**: CLAHE significantly improves vessel contrast compared to standard Histogram Equalization.
*   **Segmentation**: The combination of Gabor Filtering and Adaptive Thresholding successfully isolates the vascular network, even in the presence of noise.

### Machine Learning
*   **Accuracy**: The VGG16-based model achieves high accuracy in distinguishing Normal vs. Diabetic retinas.
*   **Evaluation**: The training script generates a **Confusion Matrix** and **Classification Report** to verify performance across all classes.

---

## 📚 References
1.  Dai, Peishan, et al. "Retinal Fundus Image Enhancement Using the Normalized Convolution and Noise Removing." International Journal of Biomedical Imaging, 2016.
2.  M.M. Fraz, et al. "Blood vessel segmentation methodologies in retinal images – A survey." Computer Methods and Programs in Biomedicine, 2012.
3.  Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv preprint arXiv:1409.1556.

---

**Author**: Janeshvar Sivakumar
**Institution**: Sri Sivasubramaniya Nadar College of Engineering
