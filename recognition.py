import numpy as np
import tensorflow as tf
import time
import cv2
from firebase_config import db, storage
from encoder import encoder
from utils import download_image, detect_and_resize_face, augment_image
from sklearn.cluster import DBSCAN

# Load Encoder Model
encoder_model = encoder  # Load the pre-trained encoder model

# Generate embeddings from images
def generate_embedding(image_paths):
    embeddings = []
    for image_path in image_paths:
        try:
            image = cv2.imread(image_path)
            face_image = detect_and_resize_face(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            augmented_images = augment_image(face_image)

            for aug_image in augmented_images:
                preprocessed_image = tf.keras.applications.inception_v3.preprocess_input(np.expand_dims(aug_image, axis=0))
                embeddings.append(encoder_model.predict(preprocessed_image))
        except ValueError as e:
            print(f"Skipping image {image_path}: {e}")
            continue

    # Cluster and compute mean embedding
    if embeddings:
        embedding_array = np.vstack(embeddings)
        dbscan = DBSCAN(eps=0.5, min_samples=5).fit(embedding_array)
        core_embeddings = embedding_array[dbscan.labels_ != -1]
        if core_embeddings.size > 0:
            return np.mean(core_embeddings, axis=0).tolist()
    return None

# Register face embeddings in Firestore
def register_face(user_id, home_id, image_paths):
    try:
        # Generate the embedding for the confirmed images
        final_embedding = generate_embedding(image_paths)
        if final_embedding is None:
            return {"error": "Failed to generate embedding"}

        # Check if user already exists in Firestore
        user_faces_ref = db.collection('userFace').where('userId', '==', user_id).where('homeId', '==', home_id).stream()
        existing_document = None
        for doc in user_faces_ref:
            existing_document = doc
            break

        # Save or update embedding in Firestore
        if existing_document:
            db.collection('userFace').document(existing_document.id).update({
                'userFace': final_embedding
            })
            return {"message": "User face updated successfully."}
        else:
            db.collection('userFace').add({
                'userId': user_id,
                'homeId': home_id,
                'userFace': final_embedding
            })
            return {"message": "User registered successfully."}
    except Exception as e:
        print(f"Error saving to Firestore: {e}")
        return {"error": "Failed to save user data"}

# Login and match face embeddings
def login_face(token , image_paths):
    # Validate token and get homeId
    device_ref = db.collection("devices").where("token", "==", token).stream()
    home_id = None
    for doc in device_ref:
        home_id = doc.to_dict().get("homeId")
        break
    if not home_id:
        print("Invalid token. Access denied.")
        return {"error": "Invalid token"}

    # Capture images and generate login embedding
    login_embedding = generate_embedding(image_paths)
    if login_embedding is None:
        print("No valid face images for login.")
        return {"error": "Failed to generate embedding for login"}

    # Retrieve stored embeddings from Firestore
    user_faces_ref = db.collection('userFace').where('homeId', '==', home_id).stream()
    closest_distance = float('inf')
    matched_user = None

    # Compare login embedding with stored embeddings
    for face in user_faces_ref:
        registered_face = np.array(face.to_dict()['userFace'])
        distance = np.linalg.norm(np.array(login_embedding) - registered_face)

        # Find closest match
        if distance < closest_distance:
            closest_distance = distance
            matched_user = face.to_dict()['userId']

    # Check if match is within threshold
    if matched_user and closest_distance < 0.6:
        print("Login successful. Door opened!" + matched_user + " " + str(closest_distance))
        return {"status": "success", "message": "Login successful"}
    else:
        print("Login failed. User not recognized." + str(closest_distance) + " " + matched_user)
        return {"status": "failed", "message": "Login failed. User not recognized"}
