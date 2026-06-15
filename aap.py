"""
AI Sign Language Detection App
Complete single-file solution - No import errors
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os
import warnings
import tempfile
import re

warnings.filterwarnings('ignore')

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Sign Language Detection",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1e2140 0%, #252a4a 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    .prediction-letter {
        font-size: 5rem;
        font-weight: bold;
        font-family: monospace;
    }
    .confidence-bar {
        height: 10px;
        border-radius: 5px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transition: width 0.3s ease;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# INITIALIZE SESSION STATE
# ============================================
if 'detected_text' not in st.session_state:
    st.session_state.detected_text = ""
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'current_confidence' not in st.session_state:
    st.session_state.current_confidence = 0.0
if 'hands_detector' not in st.session_state:
    st.session_state.hands_detector = None

# ============================================
# MEDIAPIPE LAZY LOADER
# ============================================
def get_mediapipe():
    """Lazy load MediaPipe - only when needed"""
    try:
        import mediapipe as mp
        return mp
    except ImportError:
        return None

def get_hands_detector():
    """Get or create hands detector instance"""
    if st.session_state.hands_detector is not None:
        return st.session_state.hands_detector
    
    mp = get_mediapipe()
    if mp is None:
        return None
    
    try:
        hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        st.session_state.hands_detector = hands
        return hands
    except Exception as e:
        st.warning(f"MediaPipe initialization failed: {e}")
        return None

def get_mp_drawing():
    """Get MediaPipe drawing utilities"""
    mp = get_mediapipe()
    if mp is None:
        return None, None
    return mp.solutions.drawing_utils, mp.solutions.drawing_styles

# ============================================
# HAND DETECTION FUNCTIONS
# ============================================
def extract_hand_landmarks(image):
    """Extract hand landmarks from image using MediaPipe"""
    hands = get_hands_detector()
    if hands is None:
        return None, None
    
    # Convert PIL to numpy
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to RGB (MediaPipe expects RGB)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Process
    try:
        results = hands.process(image_rgb)
        
        if results and results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            return landmarks, results
    except Exception as e:
        print(f"Error processing: {e}")
    
    return None, None

def draw_landmarks_on_image(image, results):
    """Draw hand landmarks on image"""
    mp = get_mediapipe()
    if mp is None or results is None:
        return image
    
    drawing, styles = get_mp_drawing()
    if drawing is None:
        return image
    
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image.copy()
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            drawing.draw_landmarks(
                image_rgb,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                drawing.DrawingSpec(color=(102, 126, 234), thickness=2, circle_radius=2),
                drawing.DrawingSpec(color=(118, 75, 162), thickness=2)
            )
    
    return image_rgb

def extract_landmark_array(landmarks):
    """Convert landmarks to numpy array of 63 features"""
    if landmarks is None:
        return None
    
    landmark_array = []
    for lm in landmarks.landmark:
        landmark_array.extend([lm.x, lm.y, lm.z])
    
    return np.array(landmark_array)

def recognize_hand_shape_rule_based(landmarks):
    """
    Rule-based hand shape recognition for ASL letters
    Based on finger extension patterns
    """
    if landmarks is None:
        return "?", 0.3
    
    # Extract y-coordinates for finger tips
    # MediaPipe landmark indices:
    # 4: thumb tip, 8: index tip, 12: middle tip, 16: ring tip, 20: pinky tip
    # 2: thumb IP, 5: index MCP, 9: middle MCP, 13: ring MCP, 17: pinky MCP
    
    tip_indices = [4, 8, 12, 16, 20]
    base_indices = [2, 5, 9, 13, 17]
    
    extended_fingers = []
    
    for tip_idx, base_idx in zip(tip_indices, base_indices):
        if tip_idx < len(landmarks.landmark) and base_idx < len(landmarks.landmark):
            tip_y = landmarks.landmark[tip_idx].y
            base_y = landmarks.landmark[base_idx].y
            
            if tip_y < base_y:  # Finger extended (tip above base)
                extended_fingers.append(tip_idx)
    
    extended_count = len(extended_fingers)
    
    # Map extended finger count to ASL letters
    # This is a simplified mapping
    letter_map = {
        0: ("A", 0.7),   # Fist
        1: ("D", 0.6),   # Index up
        2: ("V", 0.6),   # V sign
        3: ("W", 0.5),   # Three fingers
        4: ("B", 0.7),   # Four fingers up
        5: ("5", 0.5)    # All fingers spread
    }
    
    # Check for specific patterns
    # Thumb extended (index 4)
    if 4 in extended_fingers:
        if extended_count == 1:
            return "L", 0.7
        elif extended_count == 2 and 8 in extended_fingers:
            return "L", 0.8
    
    # Check for peace sign (index and middle only)
    if extended_count == 2 and 8 in extended_fingers and 12 in extended_fingers:
        if 16 not in extended_fingers and 20 not in extended_fingers:
            return "V", 0.8
    
    # Check for okay sign
    if 4 in extended_fingers and 8 in extended_fingers:
        if abs(landmarks.landmark[4].x - landmarks.landmark[8].x) < 0.1:
            return "O", 0.6
    
    return letter_map.get(extended_count, ("?", 0.3))

# ============================================
# TEXT-TO-SPEECH FUNCTION
# ============================================
def text_to_speech(text, lang='en'):
    """Convert text to speech using gTTS"""
    if not text or not text.strip():
        return None
    
    try:
        from gtts import gTTS
        
        tts = gTTS(text=text, lang=lang, slow=False)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        print(f"TTS error: {e}")
        return None

# ============================================
# MODEL LOADING (OPTIONAL)
# ============================================
def try_load_tensorflow_model():
    """Try to load TensorFlow model if available"""
    try:
        import tensorflow as tf
        from huggingface_hub import hf_hub_download
        
        # Try to download a small model
        model_path = hf_hub_download(
            repo_id="ColdSlim/ASL-TFLite-Edge",
            filename="asl_model.tflite",
            cache_dir="./models",
            resume=True
        )
        
        if model_path and os.path.exists(model_path):
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter, True
    except Exception as e:
        print(f"Model loading skipped: {e}")
    
    return None, False

# ============================================
# MAIN APP UI
# ============================================
st.markdown("""
<div class="main-header">
    <h1>🤟 AI Sign Language Detection</h1>
    <p>American Sign Language (ASL) recognition using MediaPipe</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Check MediaPipe status
    mp = get_mediapipe()
    if mp:
        st.success("✅ MediaPipe: Available")
    else:
        st.error("❌ MediaPipe: Not available")
    
    # Try to load ML model
    if st.button("🔄 Try Load ML Model"):
        with st.spinner("Attempting to load model..."):
            interpreter, success = try_load_tensorflow_model()
            if success:
                st.session_state.model_loaded = True
                st.session_state.interpreter = interpreter
                st.success("✅ Model loaded!")
            else:
                st.info("Using rule-based recognition")
    
    if st.session_state.model_loaded:
        st.success("Model: **Active**")
    else:
        st.info("Mode: **Rule-based recognition**")
    
    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.info("""
    1. **Upload an image** of a hand sign
    2. Click **Detect Sign**
    3. View detected letter
    4. **Add to sentence** to build words
    5. Click **Speak** to hear the sentence
    """)
    
    st.markdown("---")
    st.caption("Made with MediaPipe, Streamlit & TensorFlow")

# Main content
col1, col2 = st.columns([1, 1], gap="large")

# Column 1 - Input
with col1:
    st.markdown("## 📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Upload a clear image of an ASL hand sign"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Detect Sign", key="detect_btn", use_container_width=True):
            with st.spinner("Processing image..."):
                # Extract landmarks
                landmarks, results = extract_hand_landmarks(image)
                
                if landmarks:
                    # Draw landmarks
                    landmark_img = draw_landmarks_on_image(image, results)
                    st.image(landmark_img, caption="Hand Landmarks Detected", use_container_width=True)
                    
                    # Recognize sign
                    letter, confidence = recognize_hand_shape_rule_based(landmarks)
                    
                    st.session_state.detected_text = letter
                    st.session_state.current_confidence = confidence
                    
                    st.success(f"✅ Detected: **{letter}** (confidence: {confidence:.1%})")
                else:
                    st.warning("⚠️ No hand detected. Please try:")
                    st.markdown("""
                    - Better lighting
                    - Plain background
                    - Hand centered in frame
                    """)
    
    # Sentence Builder
    st.markdown("---")
    st.markdown("## 📝 Sentence Builder")
    
    if st.session_state.detected_text:
        st.markdown(f"""
        <div class="prediction-box">
            <div style="font-size: 0.9rem;">Last Detected</div>
            <div style="font-size: 3rem; font-weight: bold;">{st.session_state.detected_text}</div>
            <div>Confidence: {st.session_state.current_confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    col_add, col_clear = st.columns(2)
    with col_add:
        if st.button("➕ Add to Sentence", use_container_width=True):
            if st.session_state.detected_text and st.session_state.detected_text != "?":
                st.session_state.sentence += st.session_state.detected_text
                st.success(f"Added '{st.session_state.detected_text}'")
            else:
                st.warning("No valid sign detected")
    
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.sentence = ""
            st.session_state.detected_text = ""
            st.success("Cleared")

# Column 2 - Results
with col2:
    st.markdown("## 🎯 Results")
    
    # Display sentence
    st.markdown("### Current Sentence")
    sentence_display = st.text_area(
        "Sentence",
        value=st.session_state.sentence,
        height=100,
        key="sentence_display",
        disabled=True,
        label_visibility="collapsed"
    )
    
    # Speak button
    if st.button("🔊 Speak Sentence", use_container_width=True):
        if st.session_state.sentence.strip():
            with st.spinner("Generating speech..."):
                audio_file = text_to_speech(st.session_state.sentence, 'en')
                if audio_file:
                    st.audio(audio_file, format='audio/mp3')
                    st.success("Playing...")
                else:
                    st.error("Speech generation failed")
        else:
            st.warning("No sentence to speak")
    
    # Tips
    st.markdown("---")
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    - ✅ Good, even lighting
    - ✅ Plain, light-colored background
    - ✅ Hand centered in frame
    - ✅ Fingers clearly separated
    - ✅ Camera at hand level
    """)
    
    # ASL Reference
    with st.expander("📚 ASL Letter Guide"):
        st.markdown("""
        | Letter | Handshape |
        |--------|-----------|
        | **A** | Fist, thumb to side |
        | **B** | All fingers up, thumb across palm |
        | **C** | Hand shaped like C |
        | **D** | Index up, others together |
        | **L** | Index and thumb in L shape |
        | **V** | Index and middle in V shape |
        | **Y** | Thumb and pinky extended |
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8892b0; padding: 1rem;">
    <p>🤟 AI Sign Language Detection | MediaPipe + Streamlit</p>
</div>
""", unsafe_allow_html=True)
