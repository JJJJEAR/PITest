from firebase_admin import credentials, firestore, initialize_app, storage
import os

# Initialize Firebase
def initialize_firebase():
    cred = credentials.Certificate(os.path.join(os.path.dirname(__file__), "firebaseKey.json"))
    app = initialize_app(cred, {
        'databaseURL': "https://doorsystem-47cdc-default-rtdb.asia-southeast1.firebasedatabase.app",
        'storageBucket': 'gs://doorsystem-47cdc.appspot.com'
    })
    db = firestore.client()
    bucket = storage.bucket()
    return db, bucket

# Export the initialized Firestore database and Storage bucket
db, bucket = initialize_firebase()
