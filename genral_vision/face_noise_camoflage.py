import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import math 
from skimage.metrics import structural_similarity as ssim
from PIL import Image
mp_face_detection=mp.solutions.face_detection
st.title("V-information: Adversarial Noise Test")
up_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if up_file is not None:
    pili = Image.open(up_file).convert("RGB")
    original_image=np.array(pili)
    st.subheader("Original Image")
    st.image(original_image)
    # Below this everything will be done if user clicks the button Genrate image
    if st.button("Generate Image"):
        st.write("Adding adversarial noise...")
        noise_intensity=0
        ai_face_found=True
        best_camouflaged_image=original_image.copy()
        progress_bar=st.progress(0)
        # For Safe execution wrapper for the AI model
        with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
            while ai_face_found and noise_intensity < 255:
                progress_bar.progress(int((noise_intensity / 255) * 100))
                # genration for math matrix
                noise=np.random.normal(0, noise_intensity, original_image.shape)
                noise_image=np.clip(original_image + noise, 0, 255).astype(np.uint8)
                results=face_detection.process(noise_image)
                if not results.detections:
                    ai_face_found=False
                    best_camouflaged_image=noise_image
                else:
                    noise_intensity += 2
                    
        human_score=ssim(original_image, best_camouflaged_image, channel_axis=2)
        if not ai_face_found:
            st.success("Image generated successfully")
            st.info(f"Minimum Attack Threshold found:{noise_intensity}")
            st.image(best_camouflaged_image)
            st.metric(label="Human Visibility (SSIM Score)", value=f"{human_score:.3f}")
            if human_score > 0.90:
                st.write("Result: Minimal disruption to human view. Experiment Successful.")
            else:
                st.write("Result: Visible distortion. Experiment marginally successful.")
        else:
            st.error("Failed to defeat AI even at maximum noise.")