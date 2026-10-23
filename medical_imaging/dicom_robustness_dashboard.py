import streamlit as st
import numpy as np
import cv2
import pydicom
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid", context="talk", palette="deep")
plt.style.use("dark_background")
st.title("Imaging for Medical Research: AI Camouflage & Robustness Tests (Dark Mode)")
uploaded_files = st.file_uploader(
    "Upload medical scans (DICOM or images)",
    type=["dcm", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)
def load_medical_image(file):
    if file.name.endswith(".dcm"):
        ds = pydicom.dcmread(file)
        img = ds.pixel_array.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        return img.astype(np.uint8)
    else:
        return np.array(Image.open(file).convert("L"))
def apply_blur_noise(image, blur_amount, noise_intensity):
    blurred = cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)
    noise = np.random.normal(0, noise_intensity, blurred.shape)
    return np.clip(blurred + noise, 0, 255).astype(np.uint8)
def fake_ai_detector(img, blur_amount, noise_intensity):
    return not (blur_amount > 15 or noise_intensity > 20)
if uploaded_files:
    for up_file in uploaded_files:
        original_img = load_medical_image(up_file)
        st.subheader(f"Original Scan: {up_file.name}")
        st.image(original_img, clamp=True)
        if st.button(f"Run Robustness Tests for {up_file.name}"):
            st.write("Running blur/noise sweep and SSIM analysis...")
            blur_levels, noise_levels, ssim_scores, ai_results = [], [], [], []
            for blur_amount in range(3, 31, 2):
                for noise_intensity in range(0, 30, 5):
                    distorted = apply_blur_noise(original_img, blur_amount, noise_intensity)
                    ai_detected = fake_ai_detector(distorted, blur_amount, noise_intensity)
                    score = ssim(original_img, distorted)
                    blur_levels.append(blur_amount)
                    noise_levels.append(noise_intensity)
                    ssim_scores.append(score)
                    ai_results.append(ai_detected)
            #Plot 1: SSIM vs Blur/Noise
            fig1, ax1 = plt.subplots(figsize=(10,6))
            sns.scatterplot(x=blur_levels, y=ssim_scores, hue=noise_levels,
                            palette="viridis", size=noise_levels, sizes=(60,250), ax=ax1, edgecolor="white")
            ax1.set_title("Human Visibility vs Blur/Noise", fontsize=18, weight="bold", color="white")
            ax1.set_xlabel("Blur Kernel Size", fontsize=14, color="white")
            ax1.set_ylabel("SSIM Score", fontsize=14, color="white")
            ax1.tick_params(colors="white")
            ax1.legend(title="Noise Intensity", bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig1)
            #Plot 2: AI Detection vs Blur/Noise
            fig2, ax2 = plt.subplots(figsize=(10,6))
            sns.scatterplot(x=blur_levels, y=ai_results, hue=noise_levels,
                            palette="coolwarm", ax=ax2, edgecolor="white")
            ax2.set_title("AI Detection vs Blur/Noise", fontsize=18, weight="bold", color="white")
            ax2.set_xlabel("Blur Kernel Size", fontsize=14, color="white")
            ax2.set_ylabel("AI Detected (1=True, 0=False)", fontsize=14, color="white")
            ax2.tick_params(colors="white")
            ax2.legend(title="Noise Intensity", bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig2)
            #Threshold Analysis
            thresholds = []
            for b, n, s, ai in zip(blur_levels, noise_levels, ssim_scores, ai_results):
                if not ai and s > 0.85:
                    thresholds.append((b, n, s))
            if thresholds:
                best = min(thresholds, key=lambda x: x[0])
                st.success(f"Observer-Usable Threshold: Blur={best[0]}, Noise={best[1]}, SSIM={best[2]:.3f}")
            else:
                st.warning("No observer-usable threshold found (AI resisted or SSIM too low).")
            #Plot 3: Histogram of SSIM Scores
            fig3, ax3 = plt.subplots(figsize=(10,6))
            sns.histplot(ssim_scores, bins=12, kde=True, color="cyan", ax=ax3)
            ax3.set_title("Distribution of SSIM Scores", fontsize=18, weight="bold", color="white")
            ax3.set_xlabel("SSIM", fontsize=14, color="white")
            ax3.set_ylabel("Frequency", fontsize=14, color="white")
            ax3.tick_params(colors="white")
            st.pyplot(fig3)
