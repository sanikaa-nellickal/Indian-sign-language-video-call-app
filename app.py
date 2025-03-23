import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, jsonify, request, send_from_directory, redirect, url_for, flash, session
from flask_socketio import SocketIO, emit
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
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
import traceback
import time

from models import db, User, UserSettings, MeetingHistory

# Set environment variables
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_LEGACY_KERAS'] = '1'

# Initialize Flask app
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'hand_sign_recognition_secret_key'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///signconnect.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Set up login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', logger=True, engineio_logger=True)

print("="*50)
print("FLASK ROUTE REGISTRATION")
print("="*50)
print(f"App name: {app.name}")
print(f"App import name: {app.import_name}")
print(f"Static folder: {app.static_folder}")
print(f"Template folder: {app.template_folder}")
print("="*50)

# API Constants
DAILY_API_KEY = '078c49371d0577a614346b0aaf3f8110eaab590fcefc49cc8b4a3d077b112742'
DAILY_API_URL = 'https://api.daily.co/v1'
DAILY_SUBDOMAIN = 'translator'

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet += list(string.ascii_uppercase)

# Try to load TensorFlow and model
try:
    import tensorflow as tf
    print(f"Using TensorFlow version: {tf.__version__}")  # Fixed attribute name
    from tensorflow import keras
except Exception as e:
    print(f"TensorFlow import error: {e}")
    tf = None
    keras = None

print("🚀 Loading model...")
model = None
try:
    if keras:
        model = keras.models.load_model("model.h5")
        print("✓ Model loaded successfully!")
except Exception as e:
    print(f"⚠ Error loading model: {e}")
    model = None

hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Helper functions
def open_browser(url):
    import threading
    import webbrowser
    
    def _open_browser():
        time.sleep(3.0)
        # Only open the logout URL, which will redirect to login page automatically
        logout_url = "http://localhost:8080/logout"
        print(f"Opening browser at logout URL to ensure fresh session")
        webbrowser.open(logout_url)
    
    thread = threading.Thread(target=_open_browser)
    thread.daemon = True
    thread.start()
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value if max_value > 0 else 0
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

# Global variables for hand sign detection
last_prediction = None
last_emission_time = 0
stable_counter = 0
transition_detected = True
MIN_PREDICTION_CONFIDENCE = 0.7
STABLE_COUNT_REQUIRED = 3
TRANSITION_PAUSE = 1.5

def process_image(img_data):
    global last_prediction, last_emission_time, stable_counter, transition_detected
    start_time = time.time()
    current_time = time.time()
    
    if current_time - last_emission_time < TRANSITION_PAUSE:
        return None
    
    try:
        image_data = base64.b64decode(img_data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        debug_image = copy.deepcopy(image)
        
        if not results.multi_hand_landmarks:
            stable_counter = 0
            transition_detected = True
            return None
        
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            df = pd.DataFrame(pre_processed_landmark_list).transpose()
            
            if model is not None:
                try:
                    predictions = model.predict(df, verbose=0)
                    predicted_classes = np.argmax(predictions, axis=1)
                    current_prediction = alphabet[predicted_classes[0]]
                    confidence = float(predictions[0][predicted_classes[0]])
                    
                    if confidence < MIN_PREDICTION_CONFIDENCE:
                        stable_counter = 0
                        return None
                    
                    if current_prediction == last_prediction:
                        stable_counter += 1
                    else:
                        stable_counter = 1
                        last_prediction = current_prediction
                    
                    process_time = time.time() - start_time
                    print(f"Detected sign: {current_prediction} (Confidence: {confidence:.2f}, Stability: {stable_counter}/{STABLE_COUNT_REQUIRED}, Time: {process_time:.3f}s)")
                    
                    if stable_counter >= STABLE_COUNT_REQUIRED and transition_detected:
                        print(f"EMITTING SIGN: {current_prediction}")
                        last_emission_time = current_time
                        transition_detected = False
                        return current_prediction
                    
                except Exception as e:
                    print(f"Prediction error: {str(e)}")
                    traceback.print_exc()
        
        return None
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        traceback.print_exc()
        return None

# Database setup function
def create_tables():
    with app.app_context():
        db.create_all()
        print("Database tables created.")
        
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(username='admin', email='admin@signconnect.com')
            admin.set_password('adminpassword')
            db.session.add(admin)
            db.session.commit()
            print("Admin user created.")

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    print(f"Login route accessed. User authenticated: {current_user.is_authenticated}")
    if current_user.is_authenticated:
        print("User already authenticated, redirecting to dashboard")
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        print(f"Login attempt with email: {email}")
        
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            print(f"Login successful for user: {user.username}")
            user.update_last_login()
            login_user(user)
            
            next_page = request.args.get('next')
            if not next_page or url_for('login') in next_page:
                next_page = url_for('dashboard')
                
            flash('Login successful!', 'success')
            return redirect(next_page)
        else:
            print("Login failed: Invalid credentials")
            flash('Invalid email or password', 'error')
    
    print("Rendering login page")
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')
    
    if password != confirm_password:
        flash('Passwords do not match', 'error')
        return redirect(url_for('login'))
    
    existing_user = User.query.filter((User.email == email) | (User.username == username)).first()
    if existing_user:
        flash('Username or email already exists', 'error')
        return redirect(url_for('login'))
    
    try:
        user = User(email=email, username=username)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        settings = UserSettings(user_id=user.id)
        db.session.add(settings)
        db.session.commit()
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('login'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

@app.route('/static/signs/<path:filename>')
def serve_sign(filename):
    try:
        return send_from_directory('static/signs', filename)
    except Exception as e:
        print(f"Error serving sign image: {str(e)}")
        return jsonify({'error': 'Image not found'}), 404

@app.route('/')
def index():
    # Always redirect to login page from the root URL
    print("Root route accessed - forcing redirect to login page")
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    print("Dashboard route accessed")
    try:
        return render_template('index.html', user=current_user)
    except Exception as e:
        print(f"Error rendering dashboard template: {str(e)}")
        traceback.print_exc()
        return f"Error loading template: {str(e)}", 500

@app.route('/api/create-room', methods=['POST'])
@login_required
def create_room():
    try:
        room_name = f'room-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        
        print(f"Creating room: {room_name}")
        
        properties = {
            'name': room_name,
            'privacy': 'public',
            'properties': {
                'exp': int((datetime.now() + timedelta(hours=1)).timestamp()),
                'enable_chat': True,
                'enable_screenshare': True
            }
        }
        
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
                
                meeting = MeetingHistory(user_id=current_user.id, room_name=room_name)
                db.session.add(meeting)
                db.session.commit()
                
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
@login_required
def join_room(room_name):
    room_url = f'https://{DAILY_SUBDOMAIN}.daily.co/{room_name}'
    return render_template('room.html', room_name=room_name, room_url=room_url)

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        preferred_language = request.form.get('preferred_language')
        dark_mode = 'dark_mode' in request.form
        
        try:
            user_settings = UserSettings.query.filter_by(user_id=current_user.id).first()
            if user_settings:
                user_settings.preferred_language = preferred_language
                user_settings.dark_mode = dark_mode
                db.session.commit()
                flash('Settings updated successfully', 'success')
            else:
                user_settings = UserSettings(
                    user_id=current_user.id,
                    preferred_language=preferred_language,
                    dark_mode=dark_mode
                )
                db.session.add(user_settings)
                db.session.commit()
                flash('Settings created successfully', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred: {str(e)}', 'error')
    
    user_settings = UserSettings.query.filter_by(user_id=current_user.id).first()
    
    return render_template('settings.html', user=current_user, settings=user_settings)

@app.route('/history')
@login_required
def meeting_history():
    meetings = MeetingHistory.query.filter_by(user_id=current_user.id).order_by(MeetingHistory.created_on.desc()).all()
    return render_template('history.html', user=current_user, meetings=meetings)

@app.route('/check-api', methods=['GET'])
def check_api():
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

@app.route('/<path:path>')
def catch_all(path):
    print(f"Catch-all route called for path: {path}")
    return f"Path {path} not found. Available routes: /, /dashboard, /login, /profile, /join/<room_name>, /api/create-room, /check-api", 404

# Socket event handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    
@socketio.on('reset_transition')
def handle_reset_transition():
    global transition_detected
    transition_detected = True
    print("Manual transition reset received")
    emit('transition_reset', {'status': 'success'})

@socketio.on('frame')
def handle_frame(data):
    global transition_detected
    
    try:
        frame_data = data.get('frame')
        if not frame_data:
            return
            
        prediction = process_image(frame_data)
        
        if prediction:
            print(f"Emitting detected sign: {prediction}")
            emit('prediction', {'sign': prediction})
            
        if not data.get('hand_present', True):
            if not transition_detected:
                transition_detected = True
                print("Transition detected - ready for next sign")
                emit('transition_ready', {'status': 'ready'})
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")

@socketio.on('join_room')
def handle_join_room(data):
    room = data.get('room')
    if room:
        print(f"User joined room: {room}")
        socketio.join_room(room)
        socketio.emit('room_joined', {'room': room}, room=room)

# Main execution
if __name__ == '__main__':
    create_tables()
    
    os.makedirs('static/signs', exist_ok=True)
    
    import socket
    def get_ip_address():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    host = get_ip_address()
    port = 8000
    
    local_url = f"http://localhost:{port}/"  # Changed to the root URL
    
    print("\n" + "="*50)
    print(f"💻 Server is running!")
    print(f"🌐 Local URL: {local_url}")
    print(f"🌐 Network URL: http://{host}:{port}")
    print("="*50)
    print("👆 The app will open automatically in your browser")
    print("="*50 + "\n")
    
    open_browser(local_url)
    
    try:
        print(f"Attempting to start server on port {port}...")
        app.run(debug=True, host='0.0.0.0', port=port)
    except OSError as e:
        port = 5000
        print(f"Port 8080 is in use. Trying alternative port {port}...")
        local_url = f"http://localhost:{port}/"  # Changed to the root URL
        open_browser(local_url)
        app.run(debug=True, host='0.0.0.0', port=port)