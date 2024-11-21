from flask import Flask, request, jsonify, Response
import cv2
import lgpio
import time
import threading
import os
from multiprocessing import Lock
from PIL import Image
from functools import wraps
from RPLCD.i2c import CharLCD
from recognition import register_face, login_face
from firebase_config import db, bucket
import logging
import glob

# Initialize Flask app
app = Flask(__name__)

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# GPIO Setup with lgpio
chip = lgpio.gpiochip_open(0)
door_lock_pin = 25
lgpio.gpio_claim_output(chip, door_lock_pin)

# Keypad Configuration
ROW_PINS = [17, 22, 27, 10]
COL_PINS = [9, 11, 5]

KEYPAD = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    ["*", 0, "#"]
]

# LCD Setup
lcd = CharLCD('PCF8574', 0x27, cols=16, rows=2)   

lcd.clear()

number_to_text = {
    1: "Nobody Home Right Now",
    2: "Wait for a moment, please",
    3: "GET OUT OF HERE!",
    4: "Please, come in!",
}

# Locks
gpio_lock = Lock()
camera_in_use = threading.Lock()
camera_lock = threading.Lock()

# Global Variables
user_input = ""
is_collecting = False
door_open_time = None

# Logger setup
logging.basicConfig(level=logging.INFO)

def display_message(line1, line2=""):
    lcd.clear()
    lcd.write_string(f"{line1}\n{line2}")
    time.sleep(2)

def get_stored_token():
    try:
        with open('/home/pi/Desktop/PITest/deviceToken.txt', 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        logging.error("Token file not found.")
        return None

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.json.get("Token") if request.method == 'POST' else request.args.get("Token")
        if not token or token != get_stored_token():
            logging.warning("Invalid or missing token.")
            return jsonify({"error": "Invalid or missing Token"}), 403
        return f(*args, **kwargs)
    return decorated

def capture_and_upload_images(num_images):
    image_dir = './FaceImage/'
    os.makedirs(image_dir, exist_ok=True)

    # Ensure the camera is opened
    if not cap.isOpened():
        logging.info("Opening the camera...")
        cap.open(0)  # Open the camera

    image_urls = []
    try:
        for i in range(1, num_images + 1):
            image_path = f'{image_dir}image{i}.jpg'
            ret, frame = cap.read()
            if not ret:
                logging.error(f"Failed to capture image {i}")
                continue

            # Save the captured image
            cv2.imwrite(image_path, frame)
            logging.info(f'Image {i} captured.')

            # Resize and save
            with Image.open(image_path) as img:
                img = img.resize((320, 240))
                img.save(image_path)

            # Upload to Firebase
            blob = bucket.blob(f'images/image{i}.jpg')
            blob.upload_from_filename(image_path)
            blob.make_public()
            image_urls.append(blob.public_url)
            time.sleep(1)
    finally:
        # Ensure the camera is closed after use
        logging.info("Releasing the camera...")
        cap.release()
    return image_urls

@app.route('/capture', methods=['POST'])
@token_required
def capture_route():
    with gpio_lock:
        image_urls = capture_and_upload_images(7)
    return jsonify({"success": True, "images": image_urls}), 200

def control_door(state):
    global door_open_time
    if state == 'open':
        lgpio.gpio_write(chip, door_lock_pin, 1)
        door_open_time = time.time()
        display_message("Door Status:", "Opened")
        logging.info("Door opened.")
    elif state == 'close':
        lgpio.gpio_write(chip, door_lock_pin, 0)
        door_open_time = None
        display_message("Door Status:", "Closed")
        logging.info("Door closed.")

@app.route('/opendoor', methods=['POST'])
@token_required
def control_door_route():
    command = request.json.get('command')
    with gpio_lock:
        if command == 'open':
            control_door('open')
        elif command == 'close':
            control_door('close')
        else:
            return jsonify({"success": False, "message": "Invalid command."}), 400
    return jsonify({"success": True, "message": f"Door {command}ed."}), 200

def auto_close_door():
    global door_open_time
    while True:
        if door_open_time is not None:
            elapsed_time = time.time() - door_open_time
            if elapsed_time > 5:
                with gpio_lock:
                    control_door('close')
        time.sleep(10)

def setup_keypad():
    for row in ROW_PINS:
        lgpio.gpio_claim_output(chip, row)
    for col in COL_PINS:
        lgpio.gpio_claim_input(chip, col)

def read_keypad():
    global chip
    if chip is None:
        initialize_gpio()  # Reinitialize GPIO if the handle is invalid
    try:
        for i, row_pin in enumerate(ROW_PINS):
            lgpio.gpio_write(chip, row_pin, 1)
            for j, col_pin in enumerate(COL_PINS):
                if lgpio.gpio_read(chip, col_pin) == 1:
                    lgpio.gpio_write(chip, row_pin, 0)
                    while lgpio.gpio_read(chip, col_pin) == 1:
                        time.sleep(0.05)
                    return KEYPAD[i][j]
            lgpio.gpio_write(chip, row_pin, 0)
    except lgpio.error as e:
        logging.error(f"GPIO read error: {e}")
        return None


def keypad_listener():
    global user_input, is_collecting
    try:
        initialize_gpio()  
        while True:
            key = read_keypad()
            if key is not None:
                if key == "*":
                    if not is_collecting:
                        user_input = ""
                        is_collecting = True
                        logging.info("Enter 4-digit PIN:")
                        display_message("Enter PIN")
                elif key == "#":
                    capture_and_upload_images(3)
                    unlock_request()
                elif is_collecting and isinstance(key, int):
                    user_input += str(key)
                    if len(user_input) == 4:
                        display_message("PIN Entered", user_input)
                        is_collecting = False
                        pincode = get_stored_token()
                        if pincode and user_input == pincode:
                            control_door('open')
                        else:
                            display_message("Wrong PIN")
                        user_input = ""
            time.sleep(0.1)
    except Exception as e:
        logging.error(f"Error in keypad_listener: {e}")
    finally:
        close_resources(None)

def unlock_request():
    with gpio_lock:
        try:
            # Define the specific files to use
            image_dir = './FaceImage/'
            image_paths = [
                os.path.join(image_dir, f"image{i}.jpg") for i in range(1, 4)
            ]

            # Log paths being used
            logging.info(f"Using images: {image_paths}")

            # Read images from local paths
            images = []
            for path in image_paths:
                # Ensure the path is a valid string and file exists
                if not os.path.isfile(path):
                    logging.error(f"File does not exist: {path}")
                    continue

                # Explicitly ensure the path is a string and read the image
                img = cv2.imread(str(path))
                if img is None:
                    logging.error(f"Failed to load image from path: {path}")
                    continue

                images.append(img)

            if not images:
                logging.error("No valid images found for face recognition.")
                display_message("Error", "No Valid Images")
                return

            # Token retrieval
            token = get_stored_token()
            if not token:
                logging.error("Missing device token.")
                display_message("Error", "Missing Token")
                return

            # Call the face recognition function
            login_result = login_face(token, images)
            if login_result.get("status") == "success":
                control_door('open')
                display_message("Access Granted", "Door Opened")
            else:
                display_message("Access Denied")
        except Exception as e:
            logging.error(f"Error during face recognition: {e}")
            display_message("Error", "Try Again")
    time.sleep(1)
    
def unlock_request():
    with gpio_lock:
        try:
            # Define the specific files to use
            image_dir = './FaceImage/'
            image_paths = [
                os.path.join(image_dir, f"image{i}.jpg") for i in range(1, 4)
            ]

            # Log paths being used
            logging.info(f"Using images: {image_paths}")

            # Read and validate all images
            images = []
            for path in image_paths:
                # Ensure the file exists
                if not os.path.isfile(path):
                    logging.error(f"Required file does not exist: {path}")
                    display_message("Error", "Image Missing")
                    return

                # Read the image
                img = cv2.imread(str(path))
                if img is None:
                    logging.error(f"Failed to load required image from path: {path}")
                    display_message("Error", "Invalid Image")
                    return

                images.append(img)

            # Ensure all 3 images are loaded
            if len(images) != 3:
                logging.error("Not all required images are available.")
                display_message("Error", "Images Missing")
                return

            # Token retrieval
            token = get_stored_token()
            if not token:
                logging.error("Missing device token.")
                display_message("Error", "Missing Token")
                return

            # Call the face recognition function with all 3 images
            login_result = login_face(token, images)
            if login_result.get("status") == "success":
                control_door('open')
                display_message("Access Granted", "Door Opened")
            else:
                display_message("Access Denied")
        except Exception as e:
            logging.error(f"Error during face recognition: {e}")
            display_message("Error", "Try Again")
    time.sleep(1)
    
@app.route('/register', methods=['POST'])
@token_required
def register_route():
    try:
        data = request.json
        user_id = data['userId']
        home_id = data['homeId']
        
        # Capture and upload images
        image_urls = capture_and_upload_images(7)

        # Call the face recognition function
        result = register_face(user_id, home_id, image_urls)
        if result.get("status") == "success":
            return jsonify({"success": True, "message": "Face registered successfully."}), 200
        return jsonify({"success": False, "message": "Face registration failed."}), 500
    except Exception as e:
        logging.error(f"Error during registration: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        try:
            with camera_in_use:
                if not cap.isOpened():
                    logging.info("Opening the camera for video feed...")
                    cap.open(0)

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        logging.error("Failed to capture frame.")
                        continue

                    success, buffer = cv2.imencode('.jpg', frame)
                    if not success:
                        logging.error("Failed to encode frame.")
                        continue

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            logging.error(f"Error in video feed: {e}")
        finally:
            with camera_in_use:
                logging.info("Releasing camera after video feed.")
                cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.teardown_appcontext
def close_resources(exception):
    global chip
    with camera_in_use:
        if cap.isOpened():
            logging.info("Closing camera resource during app teardown...")
            cap.release()

    if chip is not None:
        try:
            logging.info("Closing GPIO chip...")
            lgpio.gpiochip_close(chip)
            chip = None  # Mark chip as closed
        except lgpio.error as e:
            logging.error(f"Error while closing GPIO chip: {e}")

def initialize_gpio():
    global chip
    if chip is None:
        try:
            logging.info("Reinitializing GPIO chip...")
            chip = lgpio.gpiochip_open(0)
            for row in ROW_PINS:
                lgpio.gpio_claim_output(chip, row)
            for col in COL_PINS:
                lgpio.gpio_claim_input(chip, col)
        except lgpio.error as e:
            logging.error(f"Failed to reinitialize GPIO chip: {e}")
            
@app.route('/textResponse', methods=['POST'])
@token_required  # Ensures the request includes a valid token
def text_response():
    try:
        # Parse the JSON request body
        data = request.json
        number = data.get('number')

        # Validate the number input
        if number is None or not isinstance(number, int) or number < 1 or number > 4:
            logging.warning("Invalid number received.")
            return jsonify({"error": "Invalid number. Provide an integer between 1 and 4."}), 400

        # Get the corresponding text for the number
        text = number_to_text.get(number, "Invalid Option")

        # Display the text on the LCD
        display_message(text)
        logging.info(f"Displayed on LCD: {text}")

        return jsonify({"success": True, "message": f"Displayed: {text}"}), 200
    except Exception as e:
        logging.error(f"Error in /textResponse: {e}")
        return jsonify({"error": "Internal server error"}), 500


try:
    threading.Thread(target=keypad_listener, daemon=True).start()
    threading.Thread(target=auto_close_door, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
finally:
    close_resources(None)



