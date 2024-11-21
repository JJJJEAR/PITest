from flask import Flask, request, jsonify, Response
import RPi.GPIO as GPIO
import time
import firebase_admin
from firebase_admin import credentials, storage, db
from picamera import PiCamera
import os
import io
import base64
import threading
import requests
from multiprocessing import Lock
from PIL import Image
from functools import wraps
from RPLCD.i2c import CharLCD
from recognition import register_face, login_face

app = Flask(__name__)

# โหลด Service Account Key และเชื่อมต่อกับ Firebase Admin SDK
cred = credentials.Certificate('/home/pi/Desktop/firebase/serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://doorsystem-47cdc-default-rtdb.asia-southeast1.firebasedatabase.app/',
    'storageBucket': 'doorsystem-47cdc.appspot.com'
})

# ตั้งค่ากล้องและ GPIO
camera = PiCamera()
camera.resolution = (640, 480)
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(25, GPIO.OUT)  

# กำหนดขา GPIO สำหรับ Keypad
ROW_PINS = [17, 6, 27, 10]  # ขา GPIO สำหรับแถว
COL_PINS = [9, 11, 5]        # ขา GPIO สำหรับคอลัมน์     

KEYPAD = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    ["*", 0, "#"]
]

# Set up I2C for LCD with address 0x27
lcd = CharLCD('PCF8574', 0x27, cols=16, rows=2)

# สร้าง Lock สำหรับควบคุมการใช้งาน GPIO
gpio_lock = Lock()
camera_lock = threading.Lock()

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

def capture_and_upload_images(num_images, camera):
    bucket = storage.bucket()
    image_urls = []
    for i in range(1, num_images + 1):
        image_path = f'/home/pi/Desktop/firebase/FaceImage/image{i}.jpg'
        camera.capture(image_path)
        print(f'Image {i} captured!')
        
        with Image.open(image_path) as img:
            img = img.resize((200, 200))
            img.save(image_path)

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
        image_urls = capture_and_upload_images(7, camera)
    return jsonify({"success": True, "images": image_urls}), 200

def control_door(state):
    global door_open_time
    if state == 'open':
        GPIO.output(25, GPIO.HIGH)
        door_open_time = time.time()  # บันทึกเวลาเปิด
        display_message("Door Status:", "Opened")
        print("Door opened")
    elif state == 'close':
        GPIO.output(25, GPIO.LOW)
        door_open_time = None  # รีเซ็ตเวลาเปิด
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
        if door_open_time is not None:  # ตรวจสอบว่าประตูเปิดอยู่
            elapsed_time = time.time() - door_open_time
            if elapsed_time > 5:  # หากเปิดเกิน 300 วินาที (5 นาที)
                with gpio_lock:
                    control_door('close')  # ปิดประตูอัตโนมัติ
        time.sleep(10)  # ตรวจสอบทุก 10 วินาที

# Trigger login request
def unlock_request():
    print("Unlock button pressed!")
    with gpio_lock:
        token = get_stored_token()
        login_result = login_face(token)
        print(login_result)
    time.sleep(1)  # Prevent multiple triggers in a short ti
    
def keypad_listener():
    global user_input, is_collecting
    GPIO.setup(ROW_PINS, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(COL_PINS, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    def read_keypad():
        for i, row_pin in enumerate(ROW_PINS):
            GPIO.output(row_pin, GPIO.HIGH)
            for j, col_pin in enumerate(COL_PINS):
                if GPIO.input(col_pin) == GPIO.HIGH:
                    GPIO.output(row_pin, GPIO.LOW)
                    # รอให้ปุ่มถูกปล่อย เพื่อดีบาวซ์
                    while GPIO.input(col_pin) == GPIO.HIGH:
                        time.sleep(0.05)
                    return KEYPAD[i][j]
            GPIO.output(row_pin, GPIO.LOW)
        return None

    def read_pincode():
        try:
            with open('/home/pi/Desktop/firebase/Pincode/pincode.txt', 'r') as file:
                return file.read().strip()
        except FileNotFoundError:
            print("Error: pincode.txt not found")
            return None

    while True:
        key = read_keypad()
        if key is not None:
            if key == "*":
                if not is_collecting:  # ตรวจสอบว่าไม่ได้อยู่ในโหมดกรอกรหัส
                    user_input = ""
                    is_collecting = True
                    print("Enter 4-digit PIN:")  # แสดงข้อความเพียงครั้งเดียว
                    display_message("Enter PIN")
            elif key == "#":
                unlock_request()
            elif is_collecting and isinstance(key, int):
                # เพิ่มดีบาวซ์หลังการกดปุ่มเพื่อหลีกเลี่ยงการอ่านซ้ำ
                user_input += str(key)
                time.sleep(0.3)  # หน่วงเวลาเพื่อป้องกันการอ่านซ้ำ
                print(user_input)
                #display_message(user_input)
                if len(user_input) == 4:
                    display_message(user_input)
                    is_collecting = False  # สิ้นสุดการเก็บข้อมูล
                    pincode = read_pincode()
                    print("pass is:",pincode)
                    print("input is:",user_input)
                    if pincode is not None:
                        if user_input == pincode:
                            print("open the door")
                            control_door('open')
                        else:
                            print("wrong password")
                            display_message("wrong password")
                    user_input = ""
        time.sleep(0.1)  # หน่วงเวลาเล็กน้อยเพื่อป้องกันการอ่านค่าซ้ำ


@app.route('/pincode', methods=['POST'])
@token_required
def save_pincode_route():
    pin_code = request.json.get('pin')
    if pin_code:
        try:
            with open('/home/pi/Desktop/firebase/Pincode/pincode.txt', 'w') as f:
                f.write(pin_code)
            return jsonify({"success": True, "message": "PIN code saved."}), 200
        except Exception as e:
            print(f"Error saving PIN code: {e}")
            return jsonify({"success": False, "message": "Failed to save PIN code."}), 500
    else:
        return jsonify({"success": False, "message": "No PIN code provided."}), 400
    
def capture_frames():
    stream = io.BytesIO()
    try:
        for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
            stream.seek(0)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + stream.read() + b'\r\n')
            stream.seek(0)
            stream.truncate()
    except Exception as e:
        print(f"Error generating frames: {e}")
            
            
@app.route('/capture_frames', methods=['GET'] )
def get_frames():
    frames = capture_frames()
    return jsonify({'frames': frames})

def generate_frames():
    stream = io.BytesIO()
    try:
        for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
            stream.seek(0)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + stream.read() + b'\r\n')
            stream.seek(0)
            stream.truncate()
    except Exception as e:
        print(f"Error generating frames: {e}")            
            
@app.route('/video_feed')      
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cam_access', methods=['POST'])
@token_required
def cam_access():
    video_feed_url = 'https://rpidoorlocksystem-ngrok-io.ngrok.io/video_feed'
    return jsonify({"success": True ,"แต่กู":"so real", "url": video_feed_url}), 200

@app.route('/register', methods=['POST'])
@token_required
def register_route():
    user_id = request.json.get('userId')
    home_id = request.json.get('homeId')
    image_urls = request.json.get('images')
    if user_id and home_id and image_urls:
        try:
            result = register_face(user_id, home_id, image_urls)
            return jsonify(result), 200
        except Exception as e:
            print(f"Error registering user: {e}")
            return jsonify({"error": "Failed to register user"}), 500
    else:
        return jsonify({"error": "Missing required fields"}), 400
    
if __name__ == "__main__":
    threading.Thread(target=keypad_listener, daemon=True).start()
    threading.Thread(target=auto_close_door, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
