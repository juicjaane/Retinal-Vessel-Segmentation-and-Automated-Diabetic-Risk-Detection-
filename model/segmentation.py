import cv2
import numpy as np
from skimage.filters import frangi, gabor_kernel
from scipy import ndimage as ndi

def extract_green_channel(img):
    return img[:, :, 1]

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def apply_gabor_filter(img, frequency=0.6):
    """
    Applies Gabor filter for vessel detection.
    """
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        kernel = np.real(gabor_kernel(frequency, theta=theta))
        kernels.append(kernel)
        
    filtered_images = []
    for kernel in kernels:
        filtered = ndi.convolve(img, kernel, mode='wrap')
        filtered_images.append(filtered)
        
    # Combine responses (e.g., max response)
    fused = np.max(filtered_images, axis=0)
    return fused

def segment_vessels(image_path, output_path=None):
    """
    Full segmentation pipeline.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image {image_path}")
        return

    # 1. Green Channel
    green = extract_green_channel(img)
    
    # 2. CLAHE
    enhanced = apply_clahe(green)
    
    # 3. Noise Reduction (Non-local Means)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # 4. Vessel Detection (using Adaptive Thresholding as a simple example, 
    #    or Gabor/Frangi if libraries allow)
    #    Here we use Adaptive Thresholding as per the README's final approach
    mask = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # 5. Morphological Operations (Opening/Closing)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    if output_path:
        cv2.imwrite(output_path, mask)
        print(f"Saved segmentation mask to {output_path}")
    
    return mask

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        input_img = sys.argv[1]
        output_img = "vessel_mask.jpg"
        segment_vessels(input_img, output_img)
    else:
        print("Usage: python segmentation.py <path_to_image>")
