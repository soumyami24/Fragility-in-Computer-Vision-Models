import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from scipy.optimize import differential_evolution
from PIL import Image
# Step1-Loading the ResNet50-taget Ai model
print("Loading the massive AI model... (This might take some time)")
#We use ResNet50,It is a heavyweight Vision AI.
model= models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval() 
#It is for testing and not in training 
# Step2-PyTorch for image prep
#ResNet expects exactly 224x224 pixels and specific color normalization.
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def process_image(img_array):
    #Converts our raw numpy array into a mathematical tensor
    img=Image.fromarray(img_array.astype('uint8'), 'RGB')
    tensor=preprocess(img).unsqueeze(0)
    return tensor

def get_prediction(img_array):
    #Give the image to the AI and gets the confidence probabilities.
    tensor=process_image(img_array)
    with torch.no_grad(): # Don't train, just look
        logits=model(tensor)
    # Convert raw AI math into a ratio (0.0 to 1.0) for easy understandin
    probs=F.softmax(logits,dim=1).numpy().squeeze()
    return probs
# ===
#The core logic for the attach
#===
def perturb_image(img_array, pixel_data):
    #Injects the single wrong pixel into the image.
    # Pixel data comes in as [X, Y, R, G, B]
    x, y, r, g, b = [int(val) for val in pixel_data]
    #Copy the image so we don't destroy the original
    perturbed_img = img_array.copy()
    # Overwrite the target pixel
    perturbed_img[y, x] = [r, g, b]
    return perturbed_img

def objective_function(pixel_data, original_img_array, true_class_id):
    #here we use scipy ,In this code scipy tries out a random pixel gives it to this function  and this function returns the AI's confidence score.
    #SciPy's goal is to drive the returned number as close to 0 as possible.
    #Apply the rogue pixel
    attack_image = perturb_image(original_img_array, pixel_data)
    #Ask the AI what it sees
    probs = get_prediction(attack_image)
    #Return the confidence of the ORIGINAL class
    confidence_in_truth = probs[true_class_id]
    return confidence_in_truth

#RUNNING THE EXPERIMENT -
if __name__ == "__main__":
    #Load your image (Make sure you have an image file here!)
    img_path = "Sample_images/dog.jpg" 
    try:
        clean_img = Image.open(img_path).convert("RGB")
        #Resize to 224x224 so the X/Y coordinates perfectly match the AI's math grid
        clean_img = clean_img.resize((224, 224))
        img_array = np.array(clean_img)
    except FileNotFoundError:
        print(f"ERROR: Put an image named {img_path} in the folder to test!")
        exit()
    # Step 1: Baseline Check
    # Let's see what the AI thinks this is BEFORE the attack
    initial_probs = get_prediction(img_array)
    true_class_id = np.argmax(initial_probs)
    initial_confidence = initial_probs[true_class_id]
    
    print(f"\n--- PRE-ATTACK ---")
    print(f"AI Category ID: {true_class_id}")
    print(f"AI Confidence: {initial_confidence * 100:.2f}%\n")

    print("--- INITIATING DIFFERENTIAL EVOLUTION ATTACK ---")
    print("SciPy is now breeding thousands of pixel combinations. This takes time...")
    #Step 2: The Attack Boundaries
    # X (0-223), Y (0-223), R (0-255), G (0-255), B (0-255)
    bounds = [(0, 223), (0, 223), (0, 255), (0, 255), (0, 255)]

    #Step 3:Use SciPy
    attack_result = differential_evolution(
        objective_function, 
        bounds, 
        args=(img_array, true_class_id), 
        maxiter=20, 
        popsize=20, 
        mutation=(0.5, 1.5),
        disp=True
    )
    #Step4: The Aftermath
    best_pixel = attack_result.x
    print(f"\n--- ATTACK COMPLETE ---")
    print(f"Worst Pixel Found at: X:{int(best_pixel[0])} Y:{int(best_pixel[1])}")
    print(f"Color Needed: RGB({int(best_pixel[2])}, {int(best_pixel[3])}, {int(best_pixel[4])})")
    #Prove it worked
    final_image = perturb_image(img_array, best_pixel)
    final_probs = get_prediction(final_image)
    final_confidence = final_probs[true_class_id]
    new_prediction_id = np.argmax(final_probs)
    print(f"\n--- THE DAMAGE ---")
    print(f"Original Confidence: {initial_confidence * 100:.2f}%")
    print(f"New Confidence: {final_confidence * 100:.2f}%")
    if true_class_id != new_prediction_id:
        print(f"VICTORY! The AI hallucinated. It now thinks the image is Category ID: {new_prediction_id}")
    else:
        print("The AI weakened, but did not break entirely. Try increasing 'maxiter' to 50.")
    #Save the corrupted image
    Image.fromarray(final_image).save("poisoned_output.png")
    print("\nSaved invisible poison image as 'poisoned_output.png'")