import os
import numpy as np
from flask import Flask, request, jsonify
from firebase_admin import credentials, firestore, initialize_app, storage
import cv2
import requests
from mtcnn.mtcnn import MTCNN
from sklearn.cluster import DBSCAN

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications import Xception

app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("./firebaseKey.json") 
initialize_app(cred, {
    'databaseURL': "https://doorsystem-47cdc-default-rtdb.asia-southeast1.firebasedatabase.app",
    'storageBucket': 'gs://doorsystem-47cdc.appspot.com'
})

# Initialize Firestore and MTCNN detector
db = firestore.client()
detector = MTCNN()

# Define the Encoder Model Architecture with the new format
def get_encoder(input_shape=(128, 128, 3)):
    pretrained_model = Xception(input_shape=input_shape, weights='imagenet', include_top=False, pooling='avg')
    for i in range(len(pretrained_model.layers) - 27):  # Freeze initial layers for transfer learning
        pretrained_model.layers[i].trainable = False
    encode_model = Sequential([
        pretrained_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ], name="Encode_Model")
    return encode_model

# Instantiate the encoder model and load weights
encoder = get_encoder()
try:
    encoder.load_weights('./encoder.h5')  # Load weights only
    print("Encoder weights loaded successfully.")
except Exception as e:
    print(f"Failed to load encoder weights: {e}")

# Updated function to download an image from Firebase Storage
def download_image(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an error for bad status
        image_np = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        return image
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image from URL {image_url}. Error: {e}")
        raise ValueError(f"Image not found or inaccessible at URL: {image_url}")
    
# Function to detect and resize face using MTCNN
def detect_and_resize_face(image, new_size=(128, 128)):
    faces = detector.detect_faces(image)
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")
    x, y, w, h = faces[0]['box']
    padding = 20
    x1, y1, x2, y2 = max(0, x-padding), max(0, y-padding), min(image.shape[1], x+w+padding), min(image.shape[0], y+h+padding)
    face_img = image[y1:y2, x1:x2]
    face_img_resized = cv2.resize(face_img, new_size, interpolation=cv2.INTER_LANCZOS4)
    return face_img_resized

# Function to augment images for robust embeddings
def augment_image(image):
    augmented_images = [image, cv2.flip(image, 1)]
    angles = [15, -15, 30, -30, 45, -45]
    for angle in angles:
        (h, w) = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented_images.append(rotated)
    return augmented_images

# Register endpoint
@app.route('/register', methods=['POST'])
def register_face():
    data = request.json
    user_id = data['userId']
    home_id = data['homeId']
    image_urls = data['imageUrls']  # List of 7 image URLs from Firebase Storage

    # Check if user already exists for this home
    existing_user = db.collection('userFace').where('userId', '==', user_id).where('homeId', '==', home_id).limit(1).stream()
    if any(existing_user):
        return jsonify({"error": "User already exists in the database."}), 400

    embeddings = []
    for image_url in image_urls:
        image = download_image(image_url)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            face_image = detect_and_resize_face(image_rgb)
        except ValueError as e:
            print(f"Skipping image at {image_url}: {e}")
            continue
        
        augmented_images = augment_image(face_image)
        for aug_image in augmented_images:
            aug_image_preprocessed = tf.keras.applications.inception_v3.preprocess_input(np.expand_dims(aug_image, axis=0))
            embedding = encoder.predict(aug_image_preprocessed)
            embeddings.append(embedding)

    if not embeddings:
        return jsonify({"error": "No valid face images for registration."}), 400

    # Stack embeddings and cluster to remove outliers
    embedding_array = np.vstack(embeddings)
    dbscan = DBSCAN(eps=0.5, min_samples=5).fit(embedding_array)
    core_embeddings = embedding_array[dbscan.labels_ != -1]

    if len(core_embeddings) == 0:
        return jsonify({"error": "No core embeddings found, unable to register face."}), 400

    # Calculate the mean embedding of the core samples
    final_embedding = np.mean(core_embeddings, axis=0).tolist()

    # Save embedding to Firestore
    try:
        db.collection('userFace').add({
            'userId': user_id,
            'homeId': home_id,
            'userFace': final_embedding
        })
        return jsonify({"message": "User registered successfully."}), 200
    except Exception as e:
        print(f"Error saving to Firestore: {e}")
        return jsonify({"error": "Failed to save user data."}), 500

# Login endpoint
@app.route('/login', methods=['POST'])
def login_face():
    data = request.json
    home_id = data['homeId']
    image_url = data['imageUrl']  # Single image URL from Firebase Storage

    # Download and process the login image
    image = download_image(image_url)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    try:
        face_image = detect_and_resize_face(image_rgb)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    augmented_images = augment_image(face_image)
    embeddings = []
    for aug_image in augmented_images:
        aug_image_preprocessed = tf.keras.applications.inception_v3.preprocess_input(np.expand_dims(aug_image, axis=0))
        embedding = encoder.predict(aug_image_preprocessed)
        embeddings.append(embedding)

    if not embeddings:
        return jsonify({"error": "No valid face images for login."}), 400

    login_embedding = np.mean(np.vstack(embeddings), axis=0)

    # Retrieve user faces with matching homeId
    user_faces_ref = db.collection('userFace').where('homeId', '==', home_id).stream()
    closest_distance = float('inf')
    matched_user = None

    for face in user_faces_ref:
        registered_face = np.array(face.to_dict()['userFace'])
        distance = np.linalg.norm(login_embedding - registered_face)
        
        # Update if a closer match is found
        if distance < closest_distance:
            closest_distance = distance
            matched_user = face.to_dict()['userId']

    # Check if the closest match is within the threshold
    if matched_user and closest_distance < 0.6:
        return jsonify({"message": f"Login successful. Welcome, {matched_user}."}), 200
    else:
        return jsonify({"error": "Face not recognized, login failed."}), 401

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
