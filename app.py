import os
import uuid
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash

import database
import face_utils

app = Flask(__name__)
app.secret_key = 'your_very_secret_key_here' # Change this for security

# Configuration
DATABASE_PATH = database.DATABASE_NAME 
REGISTERED_FACES_DIR = 'registered_faces' 

# Ensure the registered_faces directory exists
if not os.path.exists(REGISTERED_FACES_DIR):
    os.makedirs(REGISTERED_FACES_DIR)

# Initialize database
database.init_db()

def base64_to_image(base64_string):
    """
    Converts a base64 string to an OpenCV image (NumPy array).
    """
    # Remove the "data:image/jpeg;base64," part
    if "," in base64_string:
        base64_string = base64_string.split(',')[1]
    try:
        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding base64 string: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET'])
def register_page():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register_face():
    image_data_url = request.form.get('image_data')
    # name = request.form.get('name') # If you want to take name as well

    if not image_data_url:
        flash('No image received.', 'danger')
        return redirect(url_for('register_page'))

    captured_image_np = base64_to_image(image_data_url)
    if captured_image_np is None:
        flash('Error decoding the image.', 'danger')
        return redirect(url_for('register_page'))

    # 1. Get embedding from the image
    current_embedding = face_utils.get_embedding(captured_image_np)
    if current_embedding is None:
        flash('No face found in the image or error creating embedding. Please ensure your face is clearly visible.', 'danger')
        return redirect(url_for('register_page'))

   
    existing_images = os.listdir(REGISTERED_FACES_DIR)
    if existing_images:
        try:
            # Temporarily save the image so that find can use it
            temp_image_path = os.path.join(REGISTERED_FACES_DIR, "temp_capture.jpg")
            cv2.imwrite(temp_image_path, captured_image_np)

            matched_dfs = face_utils.find_closest_match(temp_image_path, REGISTERED_FACES_DIR)

            # Delete the temporary file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

            if not matched_dfs.empty:
                
                distance_col_name = f"{face_utils.MODEL_NAME}_{face_utils.DISTANCE_METRIC}"
                if distance_col_name not in matched_dfs.columns and 'distance' in matched_dfs.columns:
                    distance_col_name = 'distance' # Fallback for some versions or configurations

                if distance_col_name in matched_dfs.columns:
                    min_distance = matched_dfs[distance_col_name].min()
                    if min_distance < face_utils.SIMILARITY_THRESHOLD_ARCFACE_COSINE:
                        
                        flash('This face is already registered. (Pic already registered)', 'warning')
                        return redirect(url_for('register_page'))
                else:
                    print(f"Warning: Distance column '{distance_col_name}' or 'distance' not found in DeepFace.find results.")


        except Exception as e:
            print(f"Error during find operation in registration: {e}")
            
    image_filename = f"{uuid.uuid4().hex}.jpg"
    full_image_path = os.path.join(REGISTERED_FACES_DIR, image_filename)

    # Save the image to REGISTERED_FACES_DIR
    cv2.imwrite(full_image_path, captured_image_np)

    # Save the image name and embedding to the database
    if database.add_face(image_filename, current_embedding):
        flash('Face registered successfully.', 'success')
        return redirect(url_for('home')) # Redirect to the home page after registration
    else:
        flash('Error saving face to the database.', 'danger')
        # Delete the image if it was not saved to the database
        if os.path.exists(full_image_path):
            os.remove(full_image_path)
        return redirect(url_for('register_page'))

@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_face():
    image_data_url = request.form.get('image_data')

    if not image_data_url:
        flash('No image received.', 'danger')
        return redirect(url_for('login_page'))

    captured_image_np = base64_to_image(image_data_url)
    if captured_image_np is None:
        flash('Error decoding the image.', 'danger')
        return redirect(url_for('login_page'))

    # Check if there are any images in the registered_faces folder
    if not os.listdir(REGISTERED_FACES_DIR):
        flash('No faces registered yet. Please register first.', 'warning')
        return redirect(url_for('login_page'))

  
    temp_login_image_path = os.path.join(REGISTERED_FACES_DIR, "temp_login_capture.jpg")
    cv2.imwrite(temp_login_image_path, captured_image_np)

    try:
        matched_dfs = face_utils.find_closest_match(temp_login_image_path, REGISTERED_FACES_DIR)
    finally: # Ensure temp file is deleted
        if os.path.exists(temp_login_image_path):
            os.remove(temp_login_image_path)

    if not matched_dfs.empty:
        distance_col_name = f"{face_utils.MODEL_NAME}_{face_utils.DISTANCE_METRIC}"
        if distance_col_name not in matched_dfs.columns and 'distance' in matched_dfs.columns:
            distance_col_name = 'distance'

        if distance_col_name in matched_dfs.columns:
            min_distance_row = matched_dfs.loc[matched_dfs[distance_col_name].idxmin()]
            min_distance = min_distance_row[distance_col_name]
            identity = min_distance_row['identity'] 

            if min_distance < face_utils.SIMILARITY_THRESHOLD_ARCFACE_COSINE:
              
                matched_filename = os.path.basename(identity)
               
                return render_template('success.html', message=f'Face match successfully with {matched_filename}')
            else:
                flash('Face not matched. Distance: {:.2f}'.format(min_distance), 'danger')
                return render_template('error.html', message='Pic not match')
        else:
            flash('Distance column not found in the comparison results.', 'danger')
            return render_template('error.html', message='Error in comparison.')

    else: # matched_dfs is empty
        flash('No face matched or no face detected in the image.', 'danger')
        return render_template('error.html', message='Pic not match / No face detected in input')


if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=5000)