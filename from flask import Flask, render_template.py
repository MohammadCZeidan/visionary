from flask import Flask, render_template, request, redirect, url_for, session
import os
import cv2
import face_recognition
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'
face_dir = "registered_faces"
os.makedirs(face_dir, exist_ok=True)

def load_registered_faces():
    known_faces, known_names = [], []
    for filename in os.listdir(face_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(face_dir, filename)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_faces.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])
    return known_faces, known_names

known_faces, known_names = load_registered_faces()

def recognize_face(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)
        if True in matches:
            match_index = np.argmin(face_recognition.face_distance(known_faces, face_encoding))
            return known_names[match_index]
    return "Unknown"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('home'))
    return render_template('dashboard.html', user=session['user'])

@app.route('/login', methods=['POST'])
def login():
    file = request.files['image']
    if file:
        file_path = "temp.jpg"
        file.save(file_path)
        frame = cv2.imread(file_path)
        name = recognize_face(frame)
        os.remove(file_path)
        if name != "Unknown":
            session['user'] = name
            return redirect(url_for('dashboard'))
    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
