import numpy as np
import tensorflow as tf
import time
import cv2
from firebase_config import db, storage
from encoder import encoder
from utils import download_image, detect_and_resize_face, augment_image , detect_screen_glare
from sklearn.cluster import DBSCAN
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
def register_face(user_id, home_id, image_urls):

    if not image_urls or not isinstance(image_urls, list):
        logging.error("No image URLs provided or invalid format.")
        return {"error": "No image URLs provided or invalid format."}
     
    embeddings = [] 
    for image_url in image_urls: 
        try:
            # Download and process image
            image = download_image(image_url)
            face_image = detect_and_resize_face(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            augmented_images = augment_image(face_image)
                
            # Generate embeddings
            for aug_image in augmented_images:
                preprocessed_image = tf.keras.applications.inception_v3.preprocess_input(np.expand_dims(aug_image, axis=0))
                embeddings.append(encoder.predict(preprocessed_image))
        except ValueError as e:
            logging.warning(f"Skipping image due to error: {e}")
            continue
            
    if not embeddings:
        logging.error("No valid face images for registration.")
        return {"error": "No valid face images for registration."}
        
    # Cluster embeddings to remove outliers
    embedding_array = np.vstack(embeddings)
    dbscan_model = DBSCAN(eps=0.5, min_samples=5).fit(embedding_array)
    core_samples_mask = dbscan_model.labels_ != -1
    core_embeddings = embedding_array[core_samples_mask]
        
    if core_embeddings.size == 0:
        logging.error("No core embeddings found after clustering.")
        return {"error": "No core embeddings found after clustering."}
        
    # Calculate the mean embedding of the core samples
    final_embedding = np.mean(core_embeddings, axis=0).tolist()

    # Check if user already exists in Firestore
    user_faces_ref = db.collection('userFace').where('userId', '==', user_id).where('homeId', '==', home_id).stream()
    existing_document = None
    for doc in user_faces_ref:
        existing_document = doc
        break
        
    # Save embedding to Firestore
    try:
        if existing_document:
            # Update the existing document
            db.collection('userFace').document(existing_document.id).update({
                'userFace': final_embedding
            })
            return {"message": "User face updated successfully."}
        else:
            # Create a new document if no existing entry is found
            db.collection('userFace').add({
                'userId': user_id,
                'homeId': home_id,
                'userFace': final_embedding
            })
            return {"message": "User registered successfully."}
        
    except Exception as e:
        logging.error(f"Error during registration: {e}")
        return {"error": "Error during registration."}
       
# Login using face recognition
def login_face(token, images):
    try:
        # Validate token and retrieve homeId
        device_ref = db.collection("devices").where("token", "==", token).stream()
        home_id = None
        for doc in device_ref:
            home_id = doc.to_dict().get("homeId")
            break

        if not home_id:
            logging.error("Invalid token. Access denied.")
            return {"status": "failed", "message": "Invalid token"}

        # Ensure exactly 3 images are provided
        if len(images) != 3:
            logging.error(f"Expected 3 images, but received {len(images)}.")
            return {"status": "failed", "message": "Invalid number of images provided"}

        embeddings = []

        # Process each image in the list
        for image in images:
            try:
                # Detect and prepare the face image
                face_image = detect_and_resize_face(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Anti-spoofing: Check liveness
                if detect_screen_glare(face_image):
                    logging.warning("Detected screen glare, likely a spoofing attempt.")
                    return {"status": "failed", "message": "Spoofing detected"}

                augmented_images = augment_image(face_image)

                # Generate embeddings for each augmented image
                for aug_image in augmented_images:
                    preprocessed_image = tf.keras.applications.inception_v3.preprocess_input(
                        np.expand_dims(aug_image, axis=0)
                    )
                    embeddings.append(encoder.predict(preprocessed_image))
            except ValueError as e:
                logging.warning(f"Skipping image due to error: {e}")
                continue 

        # Ensure embeddings were generated
        if not embeddings:
            logging.error("No valid face images for login.")
            return {"status": "failed", "message": "No valid face images for login"}

        # Stack all embeddings into an array
        embedding_array = np.vstack(embeddings)

        # Perform DBSCAN clustering to identify core samples and remove outliers
        dbscan_model = DBSCAN(eps=0.3, min_samples=7).fit(embedding_array)
        core_samples_mask = dbscan_model.labels_ != -1
        core_embeddings = embedding_array[core_samples_mask]

        # Ensure core embeddings exist
        if len(core_embeddings) == 0:
            logging.error("No core embeddings found after clustering.")
            return {"status": "failed", "message": "No core embeddings found for login"}

        # Calculate the mean embedding of the core samples
        login_embedding = np.mean(core_embeddings, axis=0)

        # Retrieve stored user embeddings from Firestore
        user_faces_ref = db.collection('userFace').where('homeId', '==', home_id).stream()
        closest_distance = float('inf')
        matched_user = None

        # Compare login embedding with stored embeddings
        for face in user_faces_ref:
            registered_face = np.array(face.to_dict()['userFace'])
            distance = np.linalg.norm(login_embedding - registered_face)

            # Update the closest match
            if distance < closest_distance:
                closest_distance = distance
                matched_user = face.to_dict().get('userId')

        # Threshold for recognizing a match
        threshold = 0.6 
        if matched_user and closest_distance < threshold:
            logging.info(f"Login successful for user: {matched_user} with distance: {closest_distance}")
            return {
                "status": "success",
                "message": f"Login successful. Welcome, {matched_user}.",
                "distance": closest_distance,
            }
        else:
            logging.warning(f"Login failed. Closest distance: {closest_distance}. User: {matched_user}")
            return {
                "status": "failed",
                "message": "Login failed. User not recognized.",
                "distance": closest_distance,
            }

    except Exception as e:
        logging.error(f"Error during login_face processing: {e}")
        return {"status": "failed", "message": "Error during login process"}