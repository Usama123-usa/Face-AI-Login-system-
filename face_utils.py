from deepface import DeepFace
import numpy as np
import cv2

# You can try different DeepFace models and detector backends.
# 'ArcFace' generally gives good accuracy.
MODEL_NAME = 'ArcFace'
DETECTOR_BACKEND = 'retinaface' # Options: 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe'
DISTANCE_METRIC = 'cosine' # Options: 'cosine', 'euclidean', 'euclidean_l2'

def get_embedding(image_np):
    """
    Gets the face embedding from an image (NumPy array).
    Returns: Embedding (list) or None if no face is found.
    """
    try:
        # DeepFace.represent expects a BGR numpy array or an image path.
        # Our image from the webcam via OpenCV is already in BGR format.
        embedding_objs = DeepFace.represent(
            img_path=image_np,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True # Ensure a face is detected
        )
        if embedding_objs and len(embedding_objs) > 0:
            return embedding_objs[0]['embedding'] # Return embedding of the first detected face
        return None
    except ValueError as ve: # Typically "Face could not be detected"
        print(f"Face detection error: {ve}")
        return None
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return None

def find_closest_match(image_np, db_path):
    """
    Compares the given image with images present in db_path.
    Returns: DataFrame of matching images or an empty list if no match is found or an error occurs.
    """
    try:
        # DeepFace.find requires img_path to be a file path or numpy array.
        # db_path should be a folder containing images.
        dfs = DeepFace.find(
            img_path=image_np,
            db_path=db_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=True, # Ensure a face is detected in the input image
            silent=True # Suppress console output from DeepFace.find
        )
        # dfs is a list of dataframes, one for each detected face in img_path.
        # For a single face input, dfs[0] will be the dataframe of matches.
        if dfs and len(dfs) > 0:
            return dfs[0]
        return [] # Return empty list if no faces were processed
    except ValueError as ve:
        print(f"Face detection/finding error: {ve}")
        return []
    except Exception as e:
        print(f"Error in find_closest_match: {e}")
        return []

# For ArcFace, the threshold for 'cosine' distance is typically around 0.68.
# Lower distance means higher similarity.
# You might need to adjust this threshold according to your requirements.
SIMILARITY_THRESHOLD_ARCFACE_COSINE = 0.68