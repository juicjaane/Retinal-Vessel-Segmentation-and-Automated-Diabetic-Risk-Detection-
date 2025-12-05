import cv2
import numpy as np
import tensorflow as tf

def crop_image_from_gray(img, tol=7):
    """
    Crops the black borders from the fundus image.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def preprocess_input_image(img):
    """
    Custom preprocessing function to be passed to ImageDataGenerator.
    1. Crop black borders
    2. Resize to 224x224 (handled by generator usually, but good to be explicit if used standalone)
    3. VGG16 preprocessing
    """
    img = img.astype('uint8')
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (224, 224))
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img

def apply_clahe(img):
    """
    Applies CLAHE to the green channel of the image.
    """
    if len(img.shape) == 3:
        green_channel = img[:, :, 1]
    else:
        green_channel = img
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_green = clahe.apply(green_channel)
    return enhanced_green
