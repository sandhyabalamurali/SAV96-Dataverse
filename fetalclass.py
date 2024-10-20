import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from flask import url_for

# Initialize and load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=3
)
model.load_state_dict(torch.load('fetal_monitoring.pt', map_location=device))
model.to(device)
model.eval()

# Preprocess the input image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    return image

# Inference function
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1)
        return prediction

# Run segmentation and save the output mask
def run_segmentation(image_path, output_image_name='output_mask.png'):
    image_tensor = preprocess_image(image_path)
    output_mask = predict(model, image_tensor)
    output_mask = output_mask.squeeze().cpu().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    input_image = cv2.resize(input_image, (256, 256))
    plt.imshow(input_image, cmap='gray')
    plt.title('Input Image')

    plt.subplot(1, 2, 2)
    plt.imshow(output_mask, cmap='viridis')
    plt.title('Model Prediction')

    output_directory = 'static/images/'
    os.makedirs(output_directory, exist_ok=True)
    output_image_path = os.path.join(output_directory, output_image_name)
    plt.savefig(output_image_path)

    image_url = url_for('static', filename=f'images/{output_image_name}', _external=True)
    return image_url
