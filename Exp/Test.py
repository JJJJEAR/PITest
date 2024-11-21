import os
import logging
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from mtcnn.mtcnn import MTCNN
from sklearn.cluster import DBSCAN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Encoder Model
from encoder import load_encoder_model  # Assuming encoder module has load_encoder_model function
encoder = load_encoder_model('./models/encoder.h5')

def detect_and_resize_face(image_path, new_size=(128, 128)):
    detector = MTCNN()
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at {image_path} not found.")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")
    x, y, w, h = faces[0]['box']
    padding = 20
    face_img = img_rgb[max(0, y - padding):min(img.shape[0], y + h + padding),
                       max(0, x - padding):min(img.shape[1], x + w + padding)]
    return cv2.resize(face_img, new_size, interpolation=cv2.INTER_LANCZOS4)

def augment_image(image):
    augmented_images = [image]
    for angle in [15, -15, 30, -30, 45, -45]:
        M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1.0)
        augmented_images.append(cv2.warpAffine(image, M, (image.shape[1], image.shape[0])))
    for factor in [0.5, 0.75, 1.25, 1.5, 1.75]:
        augmented_images.append(cv2.convertScaleAbs(image, alpha=factor, beta=0))
    augmented_images.append(cv2.flip(image, 1))
    for kernel_size in [(3, 3), (5, 5)]:
        augmented_images.append(cv2.GaussianBlur(image, kernel_size, 0))
    return augmented_images

def register_face(face_images, user_id, user_name, encoder):
    embeddings = []
    for image_path in face_images:
        try:
            image = detect_and_resize_face(image_path)
            for aug_image in augment_image(image):
                aug_image_preprocessed = preprocess_input(np.expand_dims(aug_image, axis=0))
                embeddings.append(encoder.predict(aug_image_preprocessed))
        except ValueError as e:
            logger.warning(f"Skipping {image_path}: {e}")

    if not embeddings:
        logger.error("No valid face images for registration.")
        return

    embedding_array = np.vstack(embeddings)
    db = DBSCAN(eps=0.5, min_samples=5).fit(embedding_array)
    core_embeddings = embedding_array[db.labels_ != -1]

    if not core_embeddings.size:
        logger.error("No core embeddings found, unable to register face.")
        return

    final_embedding = np.mean(core_embeddings, axis=0)
    registered_faces_dir = os.getenv("REGISTERED_FACES_DIR", "registered_faces")
    os.makedirs(registered_faces_dir, exist_ok=True)
    np.save(f"{registered_faces_dir}/{user_id}_{user_name}.npy", final_embedding)
    logger.info(f"Face for {user_name} (ID: {user_id}) registered successfully.")

def login_face(face_images, encoder, threshold=0.6):
    embeddings = []
    for image_path in face_images:
        try:
            image = detect_and_resize_face(image_path)
            for aug_image in augment_image(image):
                aug_image_preprocessed = preprocess_input(np.expand_dims(aug_image, axis=0))
                embeddings.append(encoder.predict(aug_image_preprocessed))
        except ValueError as e:
            logger.warning(f"Skipping {image_path}: {e}")

    if not embeddings:
        logger.error("No valid face images for login.")
        return False

    login_embedding = np.mean(np.vstack(embeddings), axis=0)
    registered_faces_dir = os.getenv("REGISTERED_FACES_DIR", "registered_faces")

    if not os.path.exists(registered_faces_dir):
        logger.error("No registered faces found.")
        return False

    best_match = min(
        ((file_name, np.linalg.norm(login_embedding - np.load(os.path.join(registered_faces_dir, file_name))))
         for file_name in os.listdir(registered_faces_dir)),
        key=lambda x: x[1],
        default=(None, float('inf'))
    )

    if best_match[1] <= threshold:
        user_id, user_name = best_match[0].split('_')[0], '_'.join(best_match[0].split('_')[1:]).replace('.npy', '')
        logger.info(f"Login successful for user: {user_name} (ID: {user_id})")
        return True
    else:
        logger.info("Login failed: No matching face found with sufficient similarity.")
        return False
