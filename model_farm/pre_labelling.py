import csv
import numpy as np
import os
import sys
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.models.remove_background import BackgroundRemover
from backend.models.image_generation import ImageGenerator


input_dataset = os.path.join(os.path.dirname(__file__), "stage_1_dataset.csv")
output_dir = os.path.join(os.path.dirname(__file__), "image_bank/gen")  # Directory to save output images
stripped_dir = os.path.join(os.path.dirname(__file__), "image_bank/stripped")  # Directory to save stripped images

def generate_dataset():
    # Initialize models
    bg_remover = BackgroundRemover()
    img_generator = ImageGenerator()
    with open(input_dataset, 'r') as csvfile: # To optimize vram, do all imagen then do all rmbg
        reader = csv.reader(csvfile)
        for row in reader[1:]:  # Skip header row
            name = row[0]
            prompt = row[2]
            print(f"Generating images for item {name} prompt: {prompt}")
            
            # Generate images
            for i in range(5):
                generated_image = img_generator.generate_image(prompt)

                output_name = os.path.join(output_dir, f"{name}_{str(i)}.png")
                cv2.imwrite(output_name, generated_image)
                print(f"Image saved as: {output_name}")

    filenames = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    for i in range(len(filenames)):
        # Load the generated image
        image_path = os.path.join(output_dir, filenames[i])
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            continue

        # Remove background
        pil_image = bg_remover.remove_background(image)

        # Convert back to OpenCV format
        output_image = bg_remover.pil_to_cv2(pil_image)

        # Save the processed image
        output_name = os.path.join(stripped_dir, f"stripped_{filenames[i]}.png")
        cv2.imwrite(output_name, output_image)
        print(f"Processed image saved as: {output_name}")

if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(stripped_dir):
        os.makedirs(stripped_dir)
    
    generate_dataset()
    print("Dataset generation completed.")