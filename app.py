# Import eventlet and monkey patch at the very beginning
import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
import os
import requests
from datetime import datetime, timedelta
import json
import base64
import cv2
import numpy as np
import mediapipe as mp
import copy
import itertools
import string
import pandas as pd
import traceback  # Added for better error reporting
import time  # For timing measurements

def open_browser(url):
    """Open a browser window to the specified URL"""
    import threading
    import webbrowser
    import time
    
    def _open_browser():
        time.sleep(3.0)  # Increased delay to 3 seconds
        webbrowser.open(url)
    
    thread = threading.Thread(target=_open_browser)
    thread.daemon = True
    thread.start()

# Important: Set these environment variables before importing tensorflow
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_LEGACY_KERAS'] = '1'  # Enable legacy Keras support

# Now import TensorFlow after setting environment variables
import tensorflow as tf
print(f"Using TensorFlow version: {tf.__version__}")

# Import keras from tensorflow
from tensorflow import keras

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'hand_sign_recognition_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', logger=True, engineio_logger=True)

# Debug prints for Flask app configuration
print("="*50)
print("FLASK ROUTE REGISTRATION")
print("="*50)
print(f"App name: {app.name}")
print(f"App import name: {app.import_name}")
print(f"Static folder: {app.static_folder}")
print(f"Template folder: {app.template_folder}")
print("="*50)

# Your Daily.co settings
DAILY_API_KEY = '078c49371d0577a614346b0aaf3f8110eaab590fcefc49cc8b4a3d077b112742'
DAILY_API_URL = 'https://api.daily.co/v1'
DAILY_SUBDOMAIN = 'translator'

# IMPORTANT: Initialize MediaPipe solutions with EXACT numbers from visual.py
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Define the alphabet EXACTLY as visual.py does
alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet += list(string.ascii_uppercase)

# IMPORTANT: Load model EXACTLY as visual.py does
print("🚀 Loading model...")
try:
    model = keras.models.load_model("model.h5")
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"⚠ Error loading model: {e}")
    model = None

# IMPORTANT: Initialize hands model EXACTLY as visual.py does for perfect matching
hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# IMPORTANT: Helper functions EXACTLY as visual.py has them
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value if max_value > 0 else 0

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

# Improved sign recognition system with state tracking
last_prediction = None
last_emission_time = 0
stable_counter = 0
transition_detected = True  # Start by assuming we're ready for a new sign
MIN_PREDICTION_CONFIDENCE = 0.7  # Increased confidence threshold
STABLE_COUNT_REQUIRED = 3  # Changed from 5 to 3 - Number of consistent frames needed
TRANSITION_PAUSE = 1.5  # Time in seconds to wait after emitting a sign

def process_image(img_data):
    """Process image with improved sign transition detection"""
    global last_prediction, last_emission_time, stable_counter, transition_detected
    
    # Start timing for performance measurement
    start_time = time.time()
    current_time = time.time()
    
    # If we recently emitted a sign, check if enough time has passed for a new one
    if current_time - last_emission_time < TRANSITION_PAUSE:
        return None
    
    # Decode base64 image
    try:
        image_data = base64.b64decode(img_data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # IMPORTANT: Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)
        
        # To improve performance, optionally mark the image as not writeable
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        # Draw the hand annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        debug_image = copy.deepcopy(image)
        
        if not results.multi_hand_landmarks:
            # Reset when no hand is detected
            stable_counter = 0
            transition_detected = True  # Ready for next sign when hand returns
            return None
        
        # For each detected hand
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Calculate landmarks
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            
            # Pre-process landmarks
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            
            # Create dataframe
            df = pd.DataFrame(pre_processed_landmark_list).transpose()
            
            # Predict sign
            if model is not None:
                try:
                    # Use verbose=0
                    predictions = model.predict(df, verbose=0)
                    
                    # Get the predicted class
                    predicted_classes = np.argmax(predictions, axis=1)
                    current_prediction = alphabet[predicted_classes[0]]
                    
                    # Get confidence score
                    confidence = float(predictions[0][predicted_classes[0]])
                    
                    # Only accept predictions with sufficient confidence
                    if confidence < MIN_PREDICTION_CONFIDENCE:
                        stable_counter = 0  # Reset stability counter on low confidence
                        return None
                    
                    # Check if this is the same as last prediction for stability
                    if current_prediction == last_prediction:
                        stable_counter += 1
                    else:
                        stable_counter = 1  # Reset counter for new prediction
                        last_prediction = current_prediction
                    
                    # Log prediction details
                    process_time = time.time() - start_time
                    print(f"Detected sign: {current_prediction} (Confidence: {confidence:.2f}, Stability: {stable_counter}/{STABLE_COUNT_REQUIRED}, Time: {process_time:.3f}s)")
                    
                    # Check if we've seen the same sign consistently and we're ready for a new sign
                    if stable_counter >= STABLE_COUNT_REQUIRED and transition_detected:
                        print(f"EMITTING SIGN: {current_prediction}")
                        # Update state to indicate we've emitted this sign
                        last_emission_time = current_time
                        transition_detected = False  # Wait for transition before next emission
                        return current_prediction
                    
                except Exception as e:
                    print(f"Prediction error: {str(e)}")
                    traceback.print_exc()
        
        return None
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        traceback.print_exc()
        return None

@app.route('/static/signs/<path:filename>')
def serve_sign(filename):
    try:
        return send_from_directory('static/signs', filename)
    except Exception as e:
        print(f"Error serving sign image: {str(e)}")
        return jsonify({'error': 'Image not found'}), 404

@app.route('/')
def index():
    print("Index route called!")
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering index template: {str(e)}")
        traceback.print_exc()
        return f"Error loading template: {str(e)}", 500

@app.route('/api/create-room', methods=['POST'])
def create_room():
    try:
        # Create a unique room name
        room_name = f'room-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        
        # Debug print
        print(f"Creating room: {room_name}")
        
        # Room settings - keep it simple for free plan
        properties = {
            'name': room_name,
            'privacy': 'public',
            'properties': {
                'exp': int((datetime.now() + timedelta(hours=1)).timestamp()),
                'enable_chat': True,
                'enable_screenshare': True
            }
        }
        
        # Create the room
        headers = {
            'Authorization': f'Bearer {DAILY_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f'{DAILY_API_URL}/rooms',
            json=properties,
            headers=headers
        )
        
        if response.status_code == 200:
            try:
                room_data = json.loads(response.text)
                room_url = f'https://{DAILY_SUBDOMAIN}.daily.co/{room_name}'
                return jsonify({
                    'success': True,
                    'room_url': room_url,
                    'room_name': room_name
                })
            except json.JSONDecodeError as e:
                error_message = f"Failed to parse response JSON: {str(e)}"
                print(f"Error: {error_message}")
                print(f"Response content was: {response.text}")
                return jsonify({
                    'success': False,
                    'error': error_message
                }), 500
        else:
            error_message = f"Failed to create room. Status: {response.status_code}, Response: {response.text}"
            print(f"Error: {error_message}")
            return jsonify({
                'success': False,
                'error': error_message
            }), 500
            
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/join/<room_name>')
def join_room(room_name):
    room_url = f'https://{DAILY_SUBDOMAIN}.daily.co/{room_name}'
    return render_template('room.html', room_name=room_name, room_url=room_url)

@app.route('/check-api', methods=['GET'])
def check_api():
    """Test endpoint to check API connection"""
    try:
        headers = {
            'Authorization': f'Bearer {DAILY_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get(
            f'{DAILY_API_URL}/rooms',
            headers=headers
        )
        
        return jsonify({
            'status': response.status_code,
            'response': response.json() if response.status_code == 200 else response.text
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

# Add a catch-all route to debug any missed routes
@app.route('/<path:path>')
def catch_all(path):
    print(f"Catch-all route called for path: {path}")
    return f"Path {path} not found. Available routes: /, /join/<room_name>, /api/create-room, /check-api", 404

# Socket.IO handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    
# Endpoint to manually reset transition state
@socketio.on('reset_transition')
def handle_reset_transition():
    global transition_detected
    transition_detected = True
    print("Manual transition reset received")
    emit('transition_reset', {'status': 'success'})

# Process every frame for real-time responsiveness
@socketio.on('frame')
def handle_frame(data):
    """Process incoming video frames"""
    global transition_detected
    
    try:
        # Get the base64 encoded frame
        frame_data = data.get('frame')
        if not frame_data:
            return
            
        # Process the frame using MediaPipe and the model
        prediction = process_image(frame_data)
        
        # Emit the prediction back to the client
        if prediction:
            print(f"Emitting detected sign: {prediction}")
            emit('prediction', {'sign': prediction})
            
        # Check if hand is no longer in frame (empty frame data received)
        # This helps detect when user removes hand to prepare for next sign
        if not data.get('hand_present', True):
            if not transition_detected:
                transition_detected = True
                print("Transition detected - ready for next sign")
                emit('transition_ready', {'status': 'ready'})
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")

@socketio.on('join_room')
def handle_join_room(data):
    """Handle user joining a room"""
    room = data.get('room')
    if room:
        print(f"User joined room: {room}")
        socketio.join_room(room)
        socketio.emit('room_joined', {'room': room}, room=room)

if __name__ == '__main__':
    # Make sure the static/signs directory exists
    os.makedirs('static/signs', exist_ok=True)
    
    # Get the machine's IP address for local network access
    import socket
    def get_ip_address():
        try:
            # Get the primary IP address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            # Fallback to localhost if we can't determine IP
            return "127.0.0.1"
    
    host = get_ip_address()
    port = 5000
    alternative_port = 8080
    
    local_url = f"http://localhost:{port}"
    
    # Print a clear message with clickable links
    print("\n" + "="*50)
    print(f"💻 Server is running!")
    print(f"🌐 Local URL: {local_url}")
    print(f"🌐 Network URL: http://{host}:{port}")
    print(f"🌐 Alternative URL: http://localhost:{alternative_port} (if port 5000 fails)")
    print("="*50)
    print("👆 The app will open automatically in your browser")
    print("="*50 + "\n")
    
    # Automatically open browser to the application
    open_browser(local_url)
    
    try:
        # First try the default port
        print(f"Attempting to start server on port {port}...")
        socketio.run(app, debug=False, host='0.0.0.0', port=port)
    except OSError as e:
        # If port 5000 is in use, try the alternative port
        print(f"Port {port} is in use. Trying alternative port {alternative_port}...")
        local_url = f"http://localhost:{alternative_port}"
        open_browser(local_url)
        socketio.run(app, debug=False, host='0.0.0.0', port=alternative_port)