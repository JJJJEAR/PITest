from flask import Flask, request, jsonify, Response
import cv2
import lgpio
import time
import threading
from multiprocessing import Lock
from PIL import Image
from functools import wraps
from RPLCD.i2c import CharLCD
from recognition import register_face, login_face
from firebase_config import db, bucket  # Firebase configuration

app = Flask(__name__)

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# GPIO Setup with lgpio
chip = lgpio.gpiochip_open(0)
door_lock_pin = 25  # GPIO pin for door lock
lgpio.gpio_claim_output(chip, door_lock_pin)

# Keypad Configuration
ROW_PINS = [17, 6, 27, 10]
COL_PINS = [9, 11, 5]

KEYPAD = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    ["*", 0, "#"]
]

# LCD Setup
lcd = CharLCD('PCF8574', 0x27, cols=16, rows=2)

# Locks for GPIO and Camera
gpio_lock = Lock()
camera_lock = threading.Lock()

# Global Variables
user_input = ""
is_collecting = False
door_open_time = None


def display_message(line1, line2=""):
    lcd.clear()
    lcd.write_string(f"{line1}\n{line2}")
    time.sleep(2)


def get_stored_token():
    with open('/home/pi/Desktop/firebase/Token/deviceToken.txt', 'r') as file:
        return file.read().strip()


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.json.get("Token") if request.method == 'POST' else request.args.get("Token")
        if not token or token != get_stored_token():
            return jsonify({"error": "Invalid or missing Token"}), 403
        return f(*args, **kwargs)
    return decorated


def capture_and_upload_images(num_images):
    image_urls = []
    for i in range(1, num_images + 1):
        image_path = f'/home/pi/Desktop/firebase/FaceImage/image{i}.jpg'
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to capture image {i}")
            continue

        # Save the frame as an image
        cv2.imwrite(image_path, frame)
        print(f'Image {i} captured!')

        # Resize and save
        with Image.open(image_path) as img:
            img = img.resize((200, 200))
            img.save(image_path)

        # Upload to Firebase
        blob = bucket.blob(f'images/image{i}.jpg')
        blob.upload_from_filename(image_path)
        blob.make_public()
        image_urls.append(blob.public_url)
        time.sleep(1)
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
        print("Door opened")
    elif state == 'close':
        lgpio.gpio_write(chip, door_lock_pin, 0)
        door_open_time = None
        display_message("Door Status:", "Closed")
        print("Door closed")


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
    for i, row_pin in enumerate(ROW_PINS):
        lgpio.gpio_write(chip, row_pin, 1)
        for j, col_pin in enumerate(COL_PINS):
            if lgpio.gpio_read(chip, col_pin) == 1:
                lgpio.gpio_write(chip, row_pin, 0)
                while lgpio.gpio_read(chip, col_pin) == 1:
                    time.sleep(0.05)
                return KEYPAD[i][j]
        lgpio.gpio_write(chip, row_pin, 0)
    return None


def keypad_listener():
    global user_input, is_collecting
    setup_keypad()
    while True:
        key = read_keypad()
        if key is not None:
            if key == "*":
                if not is_collecting:
                    user_input = ""
                    is_collecting = True
                    print("Enter 4-digit PIN:")
                    display_message("Enter PIN")
            elif key == "#":
                image_urls = capture_and_upload_images(3)
                unlock_request(image_urls)
            elif is_collecting and isinstance(key, int):
                user_input += str(key)
                time.sleep(0.3)
                print(f"User input: {user_input}")
                if len(user_input) == 4:
                    display_message("PIN Entered", user_input)
                    is_collecting = False
                    # Check PIN
                    pincode = get_stored_token()  # Replace with your actual PIN retrieval logic
                    if pincode and user_input == pincode:
                        print("Correct PIN. Opening door...")
                        control_door('open')
                    else:
                        print("Incorrect PIN")
                        display_message("Wrong PIN")
                    user_input = ""
        time.sleep(0.1)


def unlock_request(image_urls):
    with gpio_lock:
        token = get_stored_token()
        try:
            login_result = login_face(token, image_urls)
            if login_result.get("success"):
                control_door('open')
            else:
                display_message("Access Denied")
        except Exception as e:
            print(f"Error: {e}")
            display_message("Error", "Try Again")
    time.sleep(1)


@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                continue
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.teardown_appcontext
def close_resources(exception):
    cap.release()
    lgpio.gpiochip_close(chip)


if __name__ == "__main__":
    threading.Thread(target=keypad_listener, daemon=True).start()
    threading.Thread(target=auto_close_door, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
