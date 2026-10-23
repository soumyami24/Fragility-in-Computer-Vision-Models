import os, zipfile
import numpy as np
import cv2
import pydicom
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
zip_file = "archive.zip"
extract_dir = "extracted_dataset"
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)
with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall(extract_dir)
print("Dataset extracted successfully!")
def load_medical_image(path):
    if path.lower().endswith(".dcm"):
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        return img.astype(np.uint8)
    else:
        return np.array(Image.open(path).convert("L"))
results = []
files = []
for root, _, filenames in os.walk(extract_dir):
    for fn in filenames:
        if fn.lower().endswith((".dcm", ".jpg", ".jpeg", ".png")):
            files.append(os.path.join(root, fn))

for idx, filepath in enumerate(files):
    original_img = load_medical_image(filepath)
    filename = os.path.basename(filepath)
    #loop
    blur_amount = 3
    noise_intensity = 0
    ai_detected = True
    best_camouflaged_image = original_img.copy()
    def fake_ai_detector(img):
        return not (blur_amount > 15 or noise_intensity > 20)
    while ai_detected and blur_amount < 99:
        blurred_img = cv2.GaussianBlur(original_img, (blur_amount, blur_amount), 0)
        noise = np.random.normal(0, noise_intensity, blurred_img.shape)
        noisy_blurred_img = np.clip(blurred_img + noise, 0, 255).astype(np.uint8)
        ai_detected = fake_ai_detector(noisy_blurred_img)
        if not ai_detected:
            best_camouflaged_image = noisy_blurred_img
        else:
            blur_amount += 2
            noise_intensity += 1
# SSIM
    human_score = ssim(original_img, best_camouflaged_image)
    results.append({
        "filename": filename,
        "ssim": human_score,
        "final_blur": blur_amount,
        "final_noise": noise_intensity,
        "ai_defeated": not ai_detected
    })
    print(f"[{idx+1}/{len(files)}] {filename} → SSIM={human_score:.3f}, Blur={blur_amount}, Noise={noise_intensity}, AI defeated={not ai_detected}")
df = pd.DataFrame(results)
df.to_csv("camouflage_results.csv", index=False)
print("\nResults saved to camouflage_results.csv")

failure_rate = df["ai_defeated"].mean()
print(f"Detector Failure Rate: {failure_rate*100:.2f}%")
# SSIM distribution
plt.figure()
sns.histplot(df["ssim"], bins=20)
plt.title("SSIM Distribution")
plt.savefig("ssim_distribution.png")
# Blur thresholds
plt.figure()
sns.boxplot(x=df["final_blur"])
plt.title("Blur Thresholds")
plt.savefig("blur_thresholds.png")
# Noise thresholds
plt.figure()
sns.boxplot(x=df["final_noise"])
plt.title("Noise Intensity Thresholds")
plt.savefig("noise_thresholds.png")
print("Plots saved: ssim_distribution.png, blur_thresholds.png, noise_thresholds.png")
