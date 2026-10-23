# Fragility in Computer Vision Models

Deep learning vision models can be broken by changes a human would never notice. This repository is my attempt to measure that problem and to show where it actually matters, especially in medical imaging.

The main idea is simple. A model can fail on an image that still looks perfectly clear to a person. I measure how clear the image still is to a human using SSIM (Structural Similarity Index), and I check whether the model still works. The gap between those two results is what this project is about.

Across my experiments, perturbations that keep SSIM around 0.85 to 0.90 are still easy for a human to read, yet they can make a model fail.

---
# How this project started
It began with a comment in one of the computer vision lectures. While discussing benchmark results, the professor said: "A model that scores 99 percent in the lab has only proven that it works in the lab." I had always treated high accuracy as proof of understanding, and that sentence made me doubt it.
I asked after class whether a model could be fooled by something a human would not even notice. The reply was a challenge: "Don't take my word for it. Try it, and see where it breaks."
My first test was small. I added a little noise to a face photo and the detector lost the face, even though the image looked normal to me. A one-pixel change flipped a classifier's prediction. From there I moved to medical scans, where aging scanners and noisy sensors raise the same risk, and where the same weakness can be used for patient privacy. That became medical cloaking.
This repository is the result of following that advice: measure the problem, test it, and be honest about the limits.
## What this project covers

| Question | Method | Folder |
|----------|--------|--------|
| How little noise does it take to make a face detector fail? | Gaussian noise added step by step against Google MediaPipe | `1_general_vision/` |
| Can changing one single pixel fool an image classifier? | Black-box Differential Evolution attack on ResNet-50 | `1_general_vision/` |
| Can a medical scan stay readable to a doctor but become unreadable to AI? | Blur and noise sweeps on DICOM scans, scored with SSIM | `2_medical_imaging/` |

---

## Why this matters

**Clinical safety.** X-ray tubes and CT sensors wear out over time. They add grain, blur and noise to scans. A radiologist can still read these images, but an AI model might not. If a model fails quietly while a human sees a clear scan, that is a real patient safety risk.

**Patient privacy.** If I can find the minimum amount of noise or blur that stops automated recognition while keeping the scan readable for a doctor, that can work as a privacy layer. I call this medical cloaking. Doctors read the scan normally, while unauthorized AI scrapers get far less useful data.

**Security and fraud.** Small texture changes, or even a single modified pixel, can mislead automated screening or triage systems without a human reviewer noticing anything. Knowing how easy this is helps people build better defenses.

---

## Repository structure

```
Fragility-in-Computer-Vision-Models/
├── 1_general_vision/
│   ├── face_noise_camouflage.py
│   └── one_pixel_resnet_attack.py
├── 2_medical_imaging/
│   ├── dicom_camouflage_generator.py
│   ├── dicom_robustness_dashboard.py
│   └── batch_dataset_evaluator.py
├── requirements.txt
└── README.md
```

---

## Part 1: General computer vision attacks

### face_noise_camouflage.py (Streamlit app)

This app adds zero-mean Gaussian noise to a face image in small steps. After every step it runs Google MediaPipe Face Detection and checks whether a face is still found. It stops as soon as the detector fails.

At that point it reports:
- the noise level where detection broke
- the SSIM between the original and the noisy image, which tells you how readable the image still is to a human
- a before and after comparison

What stands out is how large the gap is between the point where the detector fails and the point where a human can no longer recognize the face.

### one_pixel_resnet_attack.py (PyTorch + SciPy)

This is a black-box attack. It does not use gradients or any knowledge of the model's internals. It only looks at the output probabilities.

It uses Differential Evolution to search for one pixel change, described as `[x, y, R, G, B]`, that either:
- collapses the model's confidence in the correct class, or
- forces the model to confidently predict a wrong class

The target is a pre-trained ResNet-50. A 224 x 224 RGB image has around 150,000 values, and changing just one pixel is enough to cause a failure in many cases. That is what makes this attack so uncomfortable to look at.

---

## Part 2: Medical DICOM camouflage and robustness

### dicom_camouflage_generator.py (Streamlit app)

Loads raw hospital `.dcm` (DICOM) files as well as normal grayscale scans. It applies Gaussian kernel smoothing and sensor-style noise again and again until it reaches the minimum camouflage threshold. Once it is done, you can download the protected scan.

### dicom_robustness_dashboard.py (Streamlit analytics app)

Runs a grid sweep across blur kernel sizes and noise intensities, then plots the results in dark-mode Seaborn and Matplotlib charts. It includes scatter plots and SSIM distributions, and it helps locate the observer-usable threshold, which I define as `SSIM > 0.85`. Below that value a human reader starts to struggle with the image.

### batch_dataset_evaluator.py (automated pipeline)

A batch script for working with whole datasets. It:
1. extracts an archived medical dataset (`archive.zip`)
2. processes every scan automatically
3. saves statistical results to `camouflage_results.csv`
4. creates publication-ready boxplots of the thresholds

---

## Why SSIM

Pixel error measures like MSE and PSNR do not match how people actually see images. SSIM compares luminance, contrast and structure, so it is a better stand-in for the question "can a person still read this image?"

| SSIM value | What it means |
|------------|---------------|
| 1.00 | Identical images |
| 0.85 to 0.90 | Observer-usable zone, looks nearly the same to a human |
| Below 0.85 | Visible damage, human readability starts to drop |

---

## How the experiments work

```
Input image or DICOM scan
        |
        v
Apply a change (Gaussian noise, Gaussian blur, or a single pixel edit)
        |
        +--> Human check: SSIM between original and changed image
        |
        +--> Model check: does MediaPipe still find the face?
                          is the ResNet-50 prediction still correct?
        |
        v
Find where SSIM is still high but the model has already failed
```

---

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/soumyami24/Fragility-in-Computer-Vision-Models.git
cd Fragility-in-Computer-Vision-Models
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run it

Streamlit apps:

```bash
streamlit run 1_general_vision/face_noise_camouflage.py
streamlit run 2_medical_imaging/dicom_camouflage_generator.py
streamlit run 2_medical_imaging/dicom_robustness_dashboard.py
```

Scripts:

```bash
python 1_general_vision/one_pixel_resnet_attack.py
python 2_medical_imaging/batch_dataset_evaluator.py
```

For the batch evaluator, put your dataset in `archive.zip` where the script expects it. If your path is different, change it at the top of the file.

---

## Tech stack

| Area | Tools |
|------|-------|
| Deep learning | PyTorch, torchvision (ResNet-50) |
| Face detection | Google MediaPipe |
| Optimization | SciPy (Differential Evolution) |
| Medical imaging | pydicom, NumPy, OpenCV |
| Metrics | scikit-image (SSIM) |
| Plots | Matplotlib, Seaborn |
| Interface | Streamlit |

---

## Results

Add your own numbers and screenshots here after running the experiments.

| Experiment | Change applied | SSIM at failure | What happened |
|-----------|----------------|-----------------|---------------|
| Face camouflage | Gaussian noise | to fill in | MediaPipe stopped detecting the face |
| One-pixel attack | One pixel `[x, y, R, G, B]` | close to 1.00 | Prediction flipped or confidence collapsed |
| DICOM camouflage | Blur and noise | 0.85 or higher | Model output unreliable, scan still readable |

---

## Limitations

- SSIM is only a proxy for human readability. A proper reader study with radiologists would be the real test.
- The results depend on the models I tested (MediaPipe and ResNet-50). Other architectures may behave differently.
- Medical cloaking here is a research prototype. It is not a certified de-identification method and does not guarantee privacy.
- A determined attacker could try to undo the camouflage with denoising or adaptive methods.

---

## Future work

- Test Vision Transformers and medical-specific classifiers
- Add defenses such as adversarial training and denoising, and compare robustness
- Run a human observer study to validate the 0.85 SSIM threshold
- Add more realistic degradations such as motion blur, JPEG artifacts and Poisson noise
- Provide a Docker setup for a one-command demo

---

## Ethical use

This project is meant for research and defensive education. Please do not use these techniques to fool clinical systems, commit fraud or break privacy rules. Do not use real patient data without proper authorization and de-identification.

---

## References

- Su et al., One Pixel Attack for Fooling Deep Neural Networks (2019)
- Wang et al., Image Quality Assessment: From Error Visibility to Structural Similarity (2004)
- Google MediaPipe, PyTorch, pydicom and Streamlit documentation

---

