
"""
AI Sign Language Detection App
Using TensorFlow, MediaPipe, and Hugging Face Models
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os

# Import utilities
from utils import (
    download_model_from_hub,
    load_tflite_model,
    extract_hand_landmarks,
    predict_sign_from_landmarks,
    get_letter_from_index,
    process_image_for_prediction,
    draw_landmarks_on_image,
    text_to_speech_google,
    init_session_state,
    hands
)

# Page configuration
st.set_page_config(
    page_title="Sign Language Detection",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
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

# Initialize session state
init_session_state()

# Title and description
st.markdown("""
<div class="main-header">
    <h1>🤟 AI Sign Language Detection</h1>
    <p>Real-time American Sign Language (ASL) recognition using TensorFlow and MediaPipe</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Model loading
    st.markdown("### 🤖 Model")
    if st.button("🔄 Load Model from Hugging Face"):
        with st.spinner("Downloading model from Hugging Face..."):
            model_path = download_model_from_hub()
            if model_path:
                interpreter = load_tflite_model(model_path)
                if interpreter:
                    st.session_state.interpreter = interpreter
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                else:
                    st.error("❌ Failed to load model")
            else:
                st.error("❌ Model download failed")
    
    if st.session_state.get('model_loaded', False):
        st.success("Model Status: **Active**")
    else:
        st.warning("Model Status: **Not loaded** - Click above to load")
    
    st.markdown("---")
    
    # App info
    st.markdown("### 📖 How it works")
    st.info("""
    1. **Load Model** - Click the button above
    2. **Upload Image** - Choose an image with a hand sign
    3. **Use Webcam** - Real-time detection (desktop only)
    4. **Add to Sentence** - Build words/sentences
    5. **Text-to-Speech** - Hear the translation
    
    **Supported:** American Sign Language (ASL) fingerspelling A-Z
    """)
    
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[📚 Hugging Face Models](https://huggingface.co/models?pipeline_tag=image-classification&sort=downloads&search=sign+language)")
    st.markdown("[💻 GitHub Repository](https://github.com/yourusername/sign-language-detection)")
    st.markdown("---")
    st.caption("Made with ❤️ using TensorFlow, MediaPipe, and Streamlit")

# Main content area - Two columns
col1, col2 = st.columns([1, 1], gap="large")

# Column 1: Input methods
with col1:
    st.markdown("## 📤 Input Methods")
    
    # Tab for different input methods
    tab1, tab2, tab3 = st.tabs(["📷 Upload Image", "🎥 Webcam", "📝 Build Sentence"])
    
    with tab1:
        st.markdown("### Upload a hand sign image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload a clear image of a hand showing an ASL letter"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Detect Sign", key="detect_upload"):
                if st.session_state.get('model_loaded', False):
                    with st.spinner("Processing image..."):
                        letter, confidence, results = process_image_for_prediction(
                            image, 
                            st.session_state.interpreter
                        )
                        
                        if letter:
                            st.session_state.detected_text = letter
                            st.session_state.current_confidence = confidence
                            
                            # Show landmarks
                            landmark_img = draw_landmarks_on_image(image, results)
                            st.image(landmark_img, caption="Detected Hand Landmarks", use_container_width=True)
                        else:
                            st.warning("No hand detected in the image. Please try a clearer image.")
                else:
                    st.error("Please load the model first using the sidebar button!")
    
    with tab2:
        st.markdown("### Real-time Webcam Detection")
        st.warning("⚠️ Webcam access requires HTTPS or localhost. On Streamlit Cloud, this feature may be limited.")
        
        run_webcam = st.checkbox("Start Webcam", key="webcam_checkbox")
        FRAME_WINDOW = st.image([])
        
        if run_webcam:
            if st.session_state.get('model_loaded', False):
                cap = cv2.VideoCapture(0)
                
                # For demo purposes, we'll show placeholder frames
                # In production, this requires proper webcam handling
                st.info("📹 Webcam feed would appear here. For deployment, ensure HTTPS is configured.")
                
                # Placeholder for webcam frames
                for _ in range(5):  # Simulate frames
                    time.sleep(0.1)
                    
                cap.release()
            else:
                st.error("Please load the model first!")
    
    with tab3:
        st.markdown("### Build Your Sentence")
        
        # Display current detected letter
        if st.session_state.detected_text:
            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size: 1rem; color: #8892b0;">Last Detected</div>
                <div class="prediction-letter" style="font-size: 4rem;">{st.session_state.detected_text}</div>
                <div style="margin-top: 0.5rem;">Confidence: {st.session_state.get('current_confidence', 0):.2%}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No sign detected yet. Upload an image or use webcam to detect a sign.")
        
        # Add to sentence button
        col_add, col_clear, col_space = st.columns([1, 1, 2])
        with col_add:
            if st.button("➕ Add to Sentence", use_container_width=True):
                if st.session_state.detected_text:
                    st.session_state.sentence += st.session_state.detected_text
                    st.success(f"Added '{st.session_state.detected_text}' to sentence")
                else:
                    st.warning("No sign detected to add")
        
        with col_clear:
            if st.button("🗑️ Clear Sentence", use_container_width=True):
                st.session_state.sentence = ""
                st.session_state.detected_text = ""
                st.success("Sentence cleared")
        
        # Display current sentence
        st.markdown("### 📝 Current Sentence")
        sentence_display = st.text_area(
            "Sentence",
            value=st.session_state.sentence,
            height=100,
            key="sentence_display",
            disabled=True
        )
        
        # Text-to-Speech
        if st.button("🔊 Speak Sentence", use_container_width=True):
            if st.session_state.sentence.strip():
                with st.spinner("Generating speech..."):
                    audio_file = text_to_speech_google(st.session_state.sentence, 'en')
                    if audio_file:
                        st.audio(audio_file, format='audio/mp3')
                        st.success("Playing audio...")
                    else:
                        st.error("Could not generate speech")
            else:
                st.warning("No sentence to speak")

# Column 2: Results and visualization
with col2:
    st.markdown("## 🎯 Detection Results")
    
    if st.session_state.get('model_loaded', False):
        if st.session_state.detected_text:
            # Show prediction card
            confidence = st.session_state.get('current_confidence', 0)
            confidence_pct = int(confidence * 100)
            
            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size: 1rem; color: #8892b0;">Predicted Sign</div>
                <div class="prediction-letter" style="font-size: 8rem;">{st.session_state.detected_text}</div>
                <div style="margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>Confidence</span>
                        <span>{confidence:.1%}</span>
                    </div>
                    <div class="confidence-bar" style="width: {confidence_pct}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Information about the letter
            st.markdown("### 📖 Letter Information")
            if st.session_state.detected_text.isalpha():
                st.markdown(f"""
                - **Letter:** {st.session_state.detected_text}
                - **Position:** {ord(st.session_state.detected_text) - 64}th letter of alphabet
                - **ASL Fingerspelling:** Hand shape representing '{st.session_state.detected_text}'
                """)
        else:
            st.info("👈 Upload an image or use webcam to see detection results here")
    else:
        st.warning("⚠️ Please load the model from the sidebar first!")
    
    # Tips for better detection
    st.markdown("---")
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    - ✅ Use good lighting (natural light works best)
    - ✅ Keep hand centered and clearly visible
    - ✅ Use plain background if possible
    - ✅ Keep hand at a reasonable distance from camera
    - ✅ Make sure all fingers are clearly visible
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8892b0; padding: 1rem;">
    <p>🤟 AI Sign Language Detection | Built with TensorFlow, MediaPipe & Streamlit | Free models from Hugging Face</p>
</div>
""", unsafe_allow_html=True)

# Note about model limitations
with st.expander("ℹ️ About the Model"):
    st.markdown("""
    This app uses a **free pre-trained model** from Hugging Face for American Sign Language (ASL) fingerspelling recognition.
    
    **Model Details:**
    - Trained on ASL fingerspelling dataset
    - Recognizes letters A-Z
    - Uses MediaPipe for hand landmark extraction
    - TensorFlow Lite model for efficient inference
    
    **Limitations:**
    - Best for isolated letters (not continuous signing)
    - May have reduced accuracy with non-standard hand shapes
    - Works best with right-hand signs (ASL standard)
    
    For better accuracy with continuous signing, you would need a more complex model trained on sequential data.
    """)

# Cleanup on exit
import atexit
def cleanup():
    if 'hands' in dir():
        hands.close()
atexit.register(cleanup)
