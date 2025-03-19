import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_pymongo import PyMongo
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to your secure key

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb+srv://Govardhan:2508@cluster0.x5ss2.mongodb.net/your_database_name?retryWrites=true&w=majority&appName=Cluster0"
mongo = PyMongo(app)

# Flask-Login Configuration
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User Model
class User(UserMixin):
    def __init__(self, user_id, username):
        self.id = user_id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    user = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    if user:
        return User(str(user["_id"]), user["username"])
    return None

# Models Directory
model_dir = 'models'

# Load Trained Models
disease_models = {
    'Pneumonia': load_model(os.path.join(model_dir, 'pneumonia_model.h5'), compile=False),
    'TB': load_model(os.path.join(model_dir, 'TBB.h5'), compile=False),
    'COVID-19': load_model(os.path.join(model_dir, 'covid_vgg.h5'), compile=False),
    'Edema': load_model(os.path.join(model_dir, 'edama.h5'), compile=False),
}

# Upload Folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed File Types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image Classifier
def classify_disease(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    prediction = model.predict(img)
    return 'Positive' if prediction[0][0] > 0.5 else 'Negative'

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Signup Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Check if User or Email Exists
        existing_user = mongo.db.users.find_one({"$or": [{"username": username}, {"email": email}]})
        if existing_user:
            flash('Username or Email already exists!', 'danger')
            return redirect(url_for('signup'))

        mongo.db.users.insert_one({
            "username": username,
            "email": email,
            "password": hashed_password
        })

        flash('Signup successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_input = request.form['user_input']
        password = request.form['password']

        user = mongo.db.users.find_one({"$or": [{"username": user_input}, {"email": user_input}]})

        if user and check_password_hash(user["password"], password):
            login_user(User(str(user["_id"]), user["username"]))
            flash('Login successful!', 'success')
            return redirect(url_for('upload_and_display'))

        flash('Invalid Username or Email or Password!', 'danger')
        return redirect(url_for('login'))

    return render_template('login.html')

# Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

# Upload & Predict Route
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_and_display():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected!', 'danger')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected!', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            results = {}
            for disease, model in disease_models.items():
                results[disease] = classify_disease(filename, model)

            session['results'] = results  # ✅ Store results in session
            return redirect(url_for('result'))  # ✅ Redirect to result page

    return render_template('upload.html')

# Result Page
@app.route('/result')
@login_required
def result():
    results = session.get('results', {})  # ✅ Retrieve results from session
    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5002)
