
"""
Utility functions for Sign Language Detection
"""

import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download
import pickle
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize hands detector
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def download_model_from_hub():
    """
    Download a free sign language model from Hugging Face
    Using ASL Fingerspelling model - works out of the box
    """
    try:
        # Download TFLite model from Hugging Face
        model_path = hf_hub_download(
            repo_id="ColdSlim/ASL-TFLite-Edge",
            filename="asl_model.tflite",
            cache_dir="./models"
        )
        return model_path
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        return None

def load_tflite_model(model_path):
    """Load TensorFlow Lite model"""
    if model_path and os.path.exists(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

def extract_hand_landmarks(image):
    """
    Extract hand landmarks from image using MediaPipe
    
    Returns:
        landmarks: 21 points with x, y, z coordinates (63 features total)
    """
    # Convert PIL to OpenCV format if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert BGR to RGB (MediaPipe expects RGB)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Process the image
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        # Get first hand's landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract 21 landmarks × 3 coordinates (x, y, z) = 63 features
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        
        return np.array(landmarks), results
    
    return None, results

def predict_sign_from_landmarks(interpreter, landmarks):
    """
    Predict sign class from hand landmarks using TFLite model
    """
    if interpreter is None or landmarks is None:
        return None, 0.0
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare input (reshape to match model's expected shape)
    input_shape = input_details[0]['shape']
    landmarks_input = landmarks.astype(np.float32).reshape(1, -1)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], landmarks_input)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output[0])
    confidence = np.max(output[0])
    
    return predicted_class, confidence

def get_letter_from_index(index):
    """
    Map model output index to letter
    For ASL fingerspelling (A-Z)
    """
    # This mapping may vary based on the model
    # For ASL: A=0, B=1, ..., Z=25
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
               'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'nothing']
    
    if 0 <= index < len(letters):
        return letters[index]
    return "?"

def process_image_for_prediction(image, interpreter):
    """Complete pipeline: extract landmarks and predict"""
    landmarks, results = extract_hand_landmarks(image)
    
    if landmarks is not None:
        predicted_idx, confidence = predict_sign_from_landmarks(interpreter, landmarks)
        predicted_letter = get_letter_from_index(predicted_idx)
        return predicted_letter, confidence, results
    else:
        return None, 0.0, None

def draw_landmarks_on_image(image, results):
    """Draw hand landmarks on the image for visualization"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to RGB for drawing
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image.copy()
    
    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image_rgb,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    return image_rgb

def text_to_speech_google(text, lang='en'):
    """Convert text to speech using gTTS (free, no API key)"""
    from gtts import gTTS
    import tempfile
    import base64
    
    if not text:
        return None
    
    tts = gTTS(text=text, lang=lang, slow=False)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        return fp.name

def init_session_state():
    """Initialize Streamlit session state variables"""
    if 'detected_text' not in st.session_state:
        st.session_state.detected_text = ""
    if 'sentence' not in st.session_state:
        st.session_state.sentence = ""
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
