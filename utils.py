import cv2
import numpy as np
import requests
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import random

detector = MTCNN()

# Download an image from Firebase Storage
def download_image(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_np = np.frombuffer(response.content, np.uint8)
        return cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    except requests.RequestException as e:
        print(f"Failed to download image from URL {image_url}. Error: {e}")
        raise ValueError(f"Image not found or inaccessible at URL: {image_url}")

# Detect and resize face using MTCNN
def detect_and_resize_face(image, new_size=(128, 128)):
    faces = detector.detect_faces(image)
    if not faces:
        raise ValueError("No face detected in the image.")
    x, y, w, h = faces[0]['box']
    padding = 20
    x1, y1, x2, y2 = max(0, x - padding), max(0, y - padding), min(image.shape[1], x + w + padding), min(image.shape[0], y + h + padding)
    face_img = image[y1:y2, x1:x2]
    return cv2.resize(face_img, new_size, interpolation=cv2.INTER_LANCZOS4)

# Augment images for robust embeddings
def augment_image(image):
    augmented_images = [image]
    
    # Rotation angles
    angles = [15, -15, 30, -30, 45, -45]
    for angle in angles:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented_images.append(rotated)
    
    # Adjust brightness
    brightness_factors = [0.5, 0.75, 1.25, 1.5, 1.75]
    for factor in brightness_factors:
        bright = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        augmented_images.append(bright)
    
    # Flip image
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)
    
    # Gaussian blur
    for kernel_size in [(3, 3), (5, 5)]:
        blurred = cv2.GaussianBlur(image, kernel_size, 0)
        augmented_images.append(blurred)
    
    # Color Jitter (HSV Adjustment)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    h = cv2.add(h, random.randint(-10, 10))
    s = cv2.add(s, random.randint(-25, 25))
    v = cv2.add(v, random.randint(-25, 25))
    hsv_jittered = cv2.merge((h, s, v))
    color_jittered = cv2.cvtColor(hsv_jittered, cv2.COLOR_HSV2RGB)
    augmented_images.append(color_jittered)
    
    # Random Scaling
    h, w = image.shape[:2]
    scale = random.uniform(0.9, 1.1)
    scaled = cv2.resize(image, (int(w * scale), int(h * scale)))
    scaled_resized = cv2.resize(scaled, (w, h))
    augmented_images.append(scaled_resized)
    
    return augmented_images

def detect_screen_glare(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    reflection_ratio = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1])
    return reflection_ratio > 0.1  

