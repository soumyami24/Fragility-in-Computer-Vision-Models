
import streamlit as st
import numpy as np
import cv2
import pydicom
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import os
st.title("Imaging for medical research: AI Camouflage Tool")
uploaded_files = st.file_uploader("Upload medical scans",type=["dcm", "jpg", "jpeg", "png"],accept_multiple_files=True)
def load_medical_image(file):
    if file.name.endswith(".dcm"):
        ds = pydicom.dcmread(file)
        img = ds.pixel_array.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        return img.astype(np.uint8)
    else:
        return np.array(Image.open(file).convert("L"))  
if uploaded_files:
    for up_file in uploaded_files:
        original_img = load_medical_image(up_file)
        st.subheader(f"Original Scan: {up_file.name}")
        st.image(original_img, clamp=True)
        if st.button(f"Generate Camouflaged Scan for {up_file.name}"):
            st.write("Applying blur + noise in loop...")
            blur_amount = 3
            noise_intensity = 0
            ai_detected = True
            best_camouflaged_image = original_img.copy()
            progress_bar = st.progress(0)
            def fake_ai_detector(img):
                return not (blur_amount > 15 or noise_intensity > 20)
            while ai_detected and blur_amount < 99:
                progress_bar.progress(int((blur_amount / 99) * 100))
                blurred_img = cv2.GaussianBlur(original_img, (blur_amount, blur_amount), 0)
                noise = np.random.normal(0, noise_intensity, blurred_img.shape)
                noisy_blurred_img = np.clip(blurred_img + noise, 0, 255).astype(np.uint8)
                ai_detected = fake_ai_detector(noisy_blurred_img)
                if not ai_detected:
                    best_camouflaged_image = noisy_blurred_img
                else:
                    blur_amount += 2
                    noise_intensity += 1
            human_score = ssim(original_img, best_camouflaged_image)
            if not ai_detected:
                st.success("AI defeated! Camouflaged scan generated.")
                st.info(f"Final Blur Kernel: {blur_amount}, Noise Intensity: {noise_intensity}")
                st.image(best_camouflaged_image, clamp=True)
                st.metric(label="Human Visibility (SSIM)", value=f"{human_score:.3f}")
                if human_score > 0.85:
                    st.success("Minimal disruption to human view. Observer-usable dataset image.")
                else:
                    st.warning("⚠️Visible distortion. Usability reduced.")
                save_path = f"camouflaged_{up_file.name}.png"
                Image.fromarray(best_camouflaged_image).save(save_path)
                with open(save_path, "rb") as f:
                    st.download_button(
                        label="Download Camouflaged Scan",
                        data=f,
                        file_name=save_path,
                        mime="image/png"
                    )
            else:
                st.error("Failed to generate image that AI cant detect")
