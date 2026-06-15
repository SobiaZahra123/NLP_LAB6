"""
Utility functions for Sign Language Detection
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import os
import warnings
warnings.filterwarnings('ignore')

# Lazy loading for MediaPipe - import only when needed
def get_mediapipe():
    """Lazy load MediaPipe to avoid import errors"""
    try:
        import mediapipe as mp
        return mp
    except ImportError:
        return None

# Initialize MediaPipe lazily
_mp = None
_hands = None
_mp_drawing = None
_mp_drawing_styles = None

def init_mediapipe():
    """Initialize MediaPipe components"""
    global _mp, _hands, _mp_drawing, _mp_drawing_styles
    
    if _mp is None:
        _mp = get_mediapipe()
    
    if _mp is not None and _hands is None:
        _hands = _mp.solutions.hands
        _mp_drawing = _mp.solutions.drawing_utils
        _mp_drawing_styles = _mp.solutions.drawing_styles
        return _hands, _mp_drawing, _mp_drawing_styles, _mp
    
    return _hands, _mp_drawing, _mp_drawing_styles, _mp

def get_hands_detector():
    """Get or create hands detector instance"""
    hands_class, _, _, _ = init_mediapipe()
    if hands_class is not None:
        return hands_class.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return None

def download_model_from_hub():
    """
    Download a free sign language model from Hugging Face
    """
    try:
        from huggingface_hub import hf_hub_download
        
        # Try multiple model options
        model_options = [
            ("ColdSlim/ASL-TFLite-Edge", "asl_model.tflite"),
            ("NimaBoscarino/ASL-Letter-Classifier", "model.tflite"),
            ("Sayali99/Sign-Language-Detection", "sign_language_model.h5")
        ]
        
        for repo_id, filename in model_options:
            try:
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir="./models",
                    resume=True
                )
                if model_path and os.path.exists(model_path):
                    return model_path
            except:
                continue
        return None
    except Exception as e:
        st.warning(f"Model download not available: {e}")
        return None

def load_tflite_model(model_path):
    """Load TensorFlow Lite model"""
    if model_path and os.path.exists(model_path):
        try:
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            st.warning(f"Could not load TFLite model: {e}")
            return None
    return None

def extract_hand_landmarks(image):
    """
    Extract hand landmarks from image using MediaPipe
    
    Returns:
        landmarks: 21 points with x, y, z coordinates (63 features total)
        results: MediaPipe results object
    """
    # Initialize MediaPipe
    hands_detector = get_hands_detector()
    if hands_detector is None:
        return None, None
    
    # Convert PIL to OpenCV format if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert BGR to RGB (MediaPipe expects RGB)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Check if image is BGR (common from OpenCV)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Process the image
    results = hands_detector.process(image_rgb)
    
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
    
    try:
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
        confidence = float(np.max(output[0]))
        
        return predicted_class, confidence
    except Exception as e:
        st.warning(f"Prediction error: {e}")
        return None, 0.0

def get_letter_from_index(index):
    """
    Map model output index to letter
    For ASL fingerspelling (A-Z)
    """
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
               'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'nothing']
    
    if index is not None and 0 <= index < len(letters):
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
    _, _, _, mp = init_mediapipe()
    
    if mp is None or results is None:
        return image
    
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to RGB for drawing
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image.copy()
    
    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image_rgb,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )
    
    return image_rgb

def text_to_speech_google(text, lang='en'):
    """Convert text to speech using gTTS (free, no API key)"""
    if not text or not text.strip():
        return None
    
    try:
        from gtts import gTTS
        import tempfile
        
        tts = gTTS(text=text, lang=lang, slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        print(f"TTS error: {e}")
        return None

def init_session_state():
    """Initialize Streamlit session state variables"""
    if 'detected_text' not in st.session_state:
        st.session_state.detected_text = ""
    if 'sentence' not in st.session_state:
        st.session_state.sentence = ""
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'interpreter' not in st.session_state:
        st.session_state.interpreter = None
    if 'current_confidence' not in st.session_state:
        st.session_state.current_confidence = 0.0

def simple_hand_shape_recognition(landmarks):
    """
    Simple rule-based hand shape recognition when no ML model is available
    """
    if landmarks is None:
        return "?", 0.3
    
    # Extract finger states
    # Landmark indices for finger tips and bases
    finger_tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
    finger_bases = [2, 5, 9, 13, 17]
    
    extended_count = 0
    for tip, base in zip(finger_tips, finger_bases):
        tip_y = landmarks[tip*3 + 1] if len(landmarks) > tip*3 else 0
        base_y = landmarks[base*3 + 1] if len(landmarks) > base*3 else 0
        if tip_y < base_y:
            extended_count += 1
    
    # Simple mapping based on number of extended fingers
    if extended_count == 0:
        return "A", 0.6
    elif extended_count == 1:
        return "D", 0.5
    elif extended_count == 2:
        return "V", 0.5
    elif extended_count == 3:
        return "W", 0.5
    elif extended_count == 4:
        return "B", 0.6
    elif extended_count == 5:
        return "5", 0.5
    else:
        return "?", 0.3
