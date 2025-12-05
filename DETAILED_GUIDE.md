# Detailed Technical Guide: Retinal Vessel Segmentation & Diabetic Risk Detection

This guide provides an in-depth technical analysis of the project's workflow, covering both the traditional Image Processing pipeline for vessel segmentation and the Deep Learning pipeline for diabetic risk detection.

---

## Part 1: Image Processing Pipeline (Vessel Segmentation)

The project employs a robust preprocessing pipeline to enhance retinal fundus images and extract blood vessels. This section details the specific algorithms and techniques used.

### 1. Preprocessing Steps

#### 1.1 Green Channel Extraction
The green channel is extracted from the RGB image as it provides the highest contrast between blood vessels and the background retina. The red channel is often saturated, and the blue channel is noisy.
![RGB Channels](Images/rgb_channels.png)

#### 1.2 Contrast Enhancement (CLAHE)
**Contrast Limited Adaptive Histogram Equalization (CLAHE)** is applied to enhance local contrast. Unlike standard Histogram Equalization (HE) which operates globally, CLAHE operates on small regions (tiles) of the image.
*   **Mechanism**: The image is divided into 8x8 tiles. Histogram equalization is performed on each tile.
*   **Clip Limit (2.0)**: To prevent noise amplification in homogeneous areas, the histogram is clipped at a specific limit before equalization.
![CLAHE](Images/clahe.png)

#### 1.3 Noise Reduction
*   **Gaussian Blur**: Used for smoothing and unsharp masking.
*   **Median Filter**: Removes salt-and-pepper noise while preserving edges.
*   **Non-Local Means Denoising**: A sophisticated technique that averages pixels based on the similarity of their surrounding patches, effectively removing noise while preserving fine vessel structures.
![Non-local Means](Images/non_local_means.png)

### 2. Image Enhancement Techniques (Comparison)

Various techniques were explored to find the optimal enhancement strategy:

*   **Histogram Equalization (HE)**: Improves global contrast but may over-enhance background noise.
    ![Histogram Equalization](Images/histogram_eq.png)
*   **Adaptive Histogram Equalization (AHE)**: Better local contrast but can still amplify noise.
    ![Adaptive HE](Images/adaptive_he.png)
*   **Gamma Correction**: A non-linear operation ($V_{out} = V_{in}^\gamma$) used to adjust brightness.
    ![Gamma Correction](Images/gamma_correction.png)
*   **Unsharp Masking**: Enhances edges by subtracting a blurred version of the image from the original.
    ![Unsharp Masking](Images/unsharp_masking.png)

### 3. Vessel Detection Methodology

Instead of Deep Learning for segmentation, the project utilizes signal processing and morphological operations:

#### 3.1 Matched Filtering (Gabor Filters)
Uses Gabor kernels with varying orientations and scales to detect tubular structures (vessels). The filter response is maximal when the kernel aligns with a vessel segment.
![Matched Filtering](Images/matched_filtering.png)

#### 3.2 Laplacian-based Detection
Uses the Laplacian operator (second-order derivative) to detect edges (zero-crossings).
![Laplacian Detection](Images/laplacian_detection.png)

#### 3.3 Adaptive Thresholding
Converts the enhanced grayscale image into a binary mask. The threshold value is calculated for smaller regions, allowing the method to handle varying illumination conditions across the retina.
![Adaptive Threshold](Images/adaptive_threshold.png)

---

## Part 2: Machine Learning Pipeline (Diabetic Risk Detection)

For the classification task (Normal vs. Diabetic), the project uses Transfer Learning with VGG16.

### 1. Data Preprocessing for ML
*   **Black Border Removal**: A custom cropping function (`crop_image_from_gray`) removes the uninformative black borders around the fundus image, centering the region of interest.
*   **Resizing**: Images are resized to `224x224` pixels.
*   **Normalization**: Pixel values are scaled to the range `[0, 1]` or standardized using ImageNet mean/std.

### 2. Data Augmentation
To improve model generalization, `ImageDataGenerator` applies:
*   **Rotation**: +/- 20 degrees.
*   **Zoom**: +/- 20%.
*   **Horizontal Flip**: Randomly flipping images.
*   **Shift**: Width and height shifts.

### 3. Model Architecture: Transfer Learning (VGG16)
*   **Base Model**: VGG16 (pre-trained on ImageNet), weights frozen.
*   **Custom Head**:
    *   `GlobalAveragePooling2D`
    *   `Dense (128 units, ReLU)`
    *   `Dropout (0.5)`
    *   `Dense (Output, Softmax)`

---

## Part 3: Implementation Guide & How to Run

The core logic has been extracted into Python scripts located in the `model/` directory for ease of use and integration.

### Directory Structure
```
model/
├── preprocessing.py    # Image cropping and enhancement functions
├── segmentation.py     # Vessel segmentation pipeline
└── train_classifier.py # VGG16 training and evaluation script
```

### 1. Running Vessel Segmentation
To segment vessels from a single image:

1.  Open a terminal.
2.  Navigate to the `model/` directory:
    ```bash
    cd model
    ```
3.  Run the `segmentation.py` script with the path to your image:
    ```bash
    python segmentation.py "path/to/your/image.jpg"
    ```
    *   **Output**: A file named `vessel_mask.jpg` will be saved in the current directory.

### 2. Training the Diabetic Classifier
To train the Deep Learning model on your dataset:

1.  **Prepare Data**: Ensure your dataset is organized in the `datasets/` folder (e.g., `datasets/normal/`, `datasets/diabetes/`).
2.  **Configure Path**: Open `model/train_classifier.py` and check the `DATA_DIR` variable at the bottom. It defaults to `../datasets/`.
3.  **Run Training**:
    ```bash
    cd model
    python train_classifier.py
    ```
    *   **Process**: The script will load data, apply augmentation, train the VGG16 model for 20 epochs (with early stopping), and display evaluation metrics.
    *   **Output**: The trained model will be saved as `diabetic_retinopathy_model.h5`.

### 3. Using the Preprocessing Module
You can import the preprocessing functions in your own scripts:

```python
from model.preprocessing import crop_image_from_gray, apply_clahe
import cv2

img = cv2.imread('image.jpg')
cropped = crop_image_from_gray(img)
enhanced = apply_clahe(cropped)
```
