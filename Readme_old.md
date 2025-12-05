# Vessel Extraction in Retinal Fundus Images 

This project focuses on the extraction of retinal blood vessels from fundus images using traditional image processing techniques. The goal is to enhance the visibility and segmentation of blood vessels without relying on deep learning methods. These techniques are crucial for aiding the diagnosis of diabetic retinopathy and other retinal diseases.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Methodology](#methodology)
- [Features](#features)
- [Results](#results)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Folder Structure](#folder-structure)
- [Sample Outputs](#sample-outputs)
- [Contributors](#contributors)

---

## 🧾 Project Overview

This work focuses on the **enhancement and segmentation** of blood vessels from colored retinal fundus images. The main tasks include:
- Enhancing the green channel using **CLAHE (Contrast Limited Adaptive Histogram Equalization)**.
- Applying **matched filtering** for vessel detection.
- Performing **multiscale gradient computation and edge detection** using the **Canny operator**.
- Comparing various image enhancement and filtering techniques using **PSNR**, **SSIM**, and histograms.

## Dataset
The project uses the Retina Blood Vessel Segmentation Dataset from Kaggle, which contains:
- High-resolution retinal fundus images
- Binary mask annotations (vessel pixels marked as 1, background as 0)
- Diverse range of retinal pathologies
- Varying vessel widths and branching patterns

Dataset Link: [Retina Blood Vessel Dataset](https://www.kaggle.com/datasets/abdallahwagih/retina-blood-vessel)

## Methodology

### 1. Image Preprocessing
The preprocessing pipeline includes:
- Color space conversion (RGB to grayscale)
- Noise reduction
- Contrast enhancement

### 2. Image Enhancement Techniques
Various enhancement techniques were implemented and compared:

#### 2.1 Histogram-based Methods
- **Histogram Equalization (HE)**
  ![Histogram Equalization](Images/histogram_eq.png)
  - Improves global contrast
  - May over-enhance some regions

- **Adaptive Histogram Equalization (AHE)**
  ![Adaptive HE](Images/adaptive_he.png)
  - Local contrast enhancement
  - Better preservation of local details

- **Contrast Limited Adaptive Histogram Equalization (CLAHE)**
  ![CLAHE](Images/clahe.png)
  - Prevents over-amplification of noise
  - Better control over enhancement

#### 2.2 Spatial Domain Methods
- **Gamma Correction**
  ![Gamma Correction](Images/gamma_correction.png)
  - Non-linear transformation
  - Adjusts image brightness

- **Unsharp Masking**
  ![Unsharp Masking](Images/unsharp_masking.png)
  - Edge enhancement
  - Improves vessel visibility

#### 2.3 Frequency Domain Methods
- **Laplacian Filtering**
  ![Laplacian](Images/laplacian_filter.png)
  - Edge detection
  - Second-order derivative

- **Log Transformation**
  ![Log Transform](Images/log_transform.png)
  - Compresses dynamic range
  - Enhances dark regions

- **Exponential Transformation**
  ![Exponential Transform](Images/exp_transform.png)
  - Expands dynamic range
  - Enhances bright regions

### 3. Vessel Detection
Multiple approaches were implemented:

#### 3.1 Matched Filtering
![Matched Filtering](Images/matched_filtering.png)
- Template matching approach
- Detects vessel-like structures

#### 3.2 Laplacian-based Detection
![Laplacian Detection](Images/laplacian_detection.png)
- Second-order derivative
- Edge enhancement

#### 3.3 Non-local Means Filtering
![Non-local Means](Images/non_local_means.png)
- Noise reduction
- Structure preservation

#### 3.4 Adaptive Thresholding
![Adaptive Threshold](Images/adaptive_threshold.png)
- Local threshold computation
- Better handling of varying illumination

### 4. Performance Comparison
![Enhancement Comparison](Images/enhancement_comparison.png)
Comparison of different enhancement techniques showing their effects on vessel visibility and noise levels.

## Performance Metrics
The project uses several evaluation metrics:
1. IoU (Intersection over Union)
2. Dice Coefficient
3. Precision
4. Recall
5. Accuracy
6. PSNR (Peak Signal-to-Noise Ratio)
7. CII (Colorfulness Index Indicator)
8. Entropy

## Results
The final approach combines:
1. CLAHE for contrast enhancement
2. Non-local means filtering for noise reduction
3. Adaptive thresholding for vessel detection

## Future Improvements
1. Implementation of edge linking process
2. Removal of corneal structure artifacts
3. Image preprocessing to handle circumferential artifacts
4. Improved mask positioning for better evaluation metrics

## References
1. Dai, Peishan, et al. "Retinal Fundus Image Enhancement Using the Normalized Convolution and Noise Removing." International Journal of Biomedical Imaging, 2016.
2. T. Jintasuttisak and S. Intajag. "Color retinal image enhancement by Rayleigh contrast-limited adaptive histogram equalization." 2014.
3. M.M. Fraz, et al. "Blood vessel segmentation methodologies in retinal images – A survey." Computer Methods and Programs in Biomedicine, 2012.
4. A. W. Setiawan, et al. "Color retinal image enhancement using CLAHE." 2013.
5. Peng Feng, et al. "Enhancing retinal image by the Contourlet transform." Pattern Recognition Letters, 2007.
6. M. U. Akram, et al. "Retinal image blood vessel segmentation." 2009.
7. M. Saleh Miri and A. Mahloojifar. "A comparison study to evaluate retinal image enhancement techniques." 2009.
8. G. D. Joshi and J. Sivaswamy. "Colour Retinal Image Enhancement Based on Domain Knowledge." 2008.
9. N. R. Binti Sabri and H. B. Yazid. "Image Enhancement Methods For Fundus Retina Images." 2018.
10. A. Imran, et al. "Comparative Analysis of Vessel Segmentation Techniques in Retinal Images." IEEE Access, 2019.

## Authors

- Janeshvar Sivakumar 

## Institution
Sri Sivasubramaniya Nadar College of Engineering
(An Autonomous Institution, Affiliated to Anna University)
Kalavakkam – 603110

---

## 📦 Requirements

### Python Dependencies
```bash
# Core Dependencies
numpy>=1.21.0
opencv-python>=4.5.0
scikit-image>=0.18.0
matplotlib>=3.4.0
pandas>=1.3.0

# Machine Learning Dependencies
tensorflow>=2.8.0
scikit-learn>=0.24.0

# Jupyter Notebook Support
jupyter>=1.0.0
ipykernel>=6.0.0
```

### Installation
```bash
# Create a new conda environment (recommended)
conda create -n retina python=3.9
conda activate retina

# Install dependencies
pip install -r requirements.txt
```

### Hardware Requirements
- Minimum 8GB RAM
- GPU with at least 4GB VRAM (recommended for faster processing)
- 10GB free disk space for dataset and processing

### Dataset Requirements
- Retina Blood Vessel Segmentation Dataset from Kaggle
- Minimum image resolution: 224x224 pixels
- Supported formats: PNG, JPG, JPEG

