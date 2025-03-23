from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash
import os
import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///signconnect.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Define User model directly in this file
class User(db.Model):
    _tablename_ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_on = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

class UserSettings(db.Model):
    _tablename_ = 'user_settings'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    preferred_language = db.Column(db.String(10), default='en')
    dark_mode = db.Column(db.Boolean, default=False)

class MeetingHistory(db.Model):
    _tablename_ = 'meeting_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    room_name = db.Column(db.String(100), nullable=False)
    created_on = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    duration = db.Column(db.Integer, nullable=True)

def init_db():
    # Create all tables
    db.create_all()
    print("Database tables created successfully.")
    
    # Check if admin user exists
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        # Create admin user with hashed password
        admin = User(
            username='admin', 
            email='admin@signconnect.com',
            password_hash=generate_password_hash('adminpassword', method='pbkdf2:sha256')
        )
        db.session.add(admin)
        db.session.commit()
        print("Admin user created.")
        
        # Create admin settings
        admin_settings = UserSettings(user_id=admin.id)
        db.session.add(admin_settings)
        db.session.commit()
        print("Admin settings created.")
    else:
        print("Admin user already exists.")
    
    print("Database initialization complete.")

if __name__ == '__main__':
    # Check if database file exists
    db_file = 'signconnect.db'
    if os.path.exists(db_file):
        response = input(f"Database file '{db_file}' already exists. Do you want to reset it? (y/n): ")
        if response.lower() == 'y':
            os.remove(db_file)
            print(f"Database file '{db_file}' removed.")
        else:
            print("Using existing database file.")
    
    # Initialize database
    init_db()