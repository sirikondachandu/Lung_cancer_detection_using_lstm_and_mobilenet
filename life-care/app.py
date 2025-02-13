from flask import Flask, render_template, redirect, url_for, flash, request, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import os
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNet

import os

import numpy as np

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.layers import GlobalAveragePooling2D


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    mobile = db.Column(db.String(15), nullable=False)
    gender = db.Column(db.Enum('M', 'F', 'O'), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id  # Store user ID in session
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')   # Updated from 'username' to 'name'
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        age = request.form.get('age')
        gender = request.form.get('gender')
        mobile = request.form.get('mobile')

        # Validate mobile number
        if len(mobile) != 10 or not mobile.isdigit():
            flash('Mobile number must be exactly 10 digits.', 'danger')
            return render_template('login.html')

        # Check if email already exists
        if User.query.filter_by(email=email).first():
            flash('Email address already in use. Please choose a different one.', 'danger')
            return render_template('login.html')

        # Check if name (username) already exists
        if User.query.filter_by(name=name).first():
            flash('Name is already taken. Please choose a different one.', 'danger')
            return render_template('login.html')

        # Validate password
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('login.html')

        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return render_template('login.html')

        # Hash the password
        hashed_password = bcrypt.generate_password_hash(password)

        # Create a new user instance
        new_user = User(
            name=name,
            email=email,
            password=hashed_password,
            age=age,
            gender=gender,
            mobile=mobile
        )

        # Add and commit the new user to the database
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('login.html')


#============================================================================================

import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# Load the model
model = load_model('mobilenet_lstm_lung_cancer_model.h5')

# Class names
class_names = ['Benign case', 'Malignant case', 'Normal case']

def make_prediction(model, image_path):
    """
    Preprocess the image and make a prediction using the trained model.
    """
    # Load the image and preprocess it
    img = load_img(image_path, target_size=(224, 224))  # Resize to MobileNet input size
    img_array = img_to_array(img)  # Convert to a NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for MobileNet

    # Predict the class using the trained classification model
    predictions = model.predict(img_array)  # Directly predict using the model
    predicted_class_idx = np.argmax(predictions)  # Get index of the highest probability
    predicted_class = class_names[predicted_class_idx]  # Map index to class name

    return predicted_class

@app.route('/prediction', methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        myfile = request.files['file']  # Get the uploaded file
        filename = myfile.filename  # Extract filename
        upload_folder = os.path.join('static', 'img')  # Path to the upload folder
        
        # Check if the 'static/img' directory exists, if not, create it
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        # Save the file to the server
        mypath = os.path.join(upload_folder, filename)
        myfile.save(mypath)

        # Make prediction
        predicted_class = make_prediction(model, mypath)

        # Extract the filename for display
        file_name_to_display = os.path.basename(mypath)

        # Return result to the template
        return render_template('prediction.html', path=file_name_to_display, prediction=predicted_class)

    return render_template('prediction.html')

#============================================================================================

@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Remove user ID from session
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

