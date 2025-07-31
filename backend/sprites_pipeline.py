from diffusers import StableDiffusionXLPipeline
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import time
import os


# Load background removal model
rmbg_model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
rmbg_model.to('cuda')
rmbg_model.eval()

# Data settings for background removal
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def generate_image(prompt):
    image = pipe(prompt=prompt, negative_prompt=neg_prompt).images[0]
    print("PIL type:", type(image))  # Should be PIL.Image.Image
    image = image.convert("RGB")          # 🟢 Ensure RGB mode
    image_np = np.array(image)
    # print("Numpy array shape:", image_np.shape)
    # print("Numpy array type:", image_np.dtype)
    # print("Numpy array min/max:", image_np.min(), image_np.max())
    output = cv2.cvtColor(src=image_np, code=cv2.COLOR_RGB2BGR)
    return output

def remove_background(image):
    # Convert OpenCV image (BGR) back to PIL Image (RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Prepare image for background removal model
    input_images = transform_image(pil_image).unsqueeze(0).to('cuda')
    
    # Prediction
    with torch.no_grad():
        preds = rmbg_model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(pil_image.size)
    
    # Apply alpha channel (transparency) using the mask
    pil_image.putalpha(mask)
    
    # Convert back to OpenCV format with transparency
    # Note: OpenCV doesn't handle RGBA well, so we'll return PIL image
    return pil_image

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print(np.__version__)
    # to cuda
    while True:
        prompt = input("Enter your prompt: ")
        output = generate_image(prompt)
        cv2.imshow("Generated Image", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Ask if user wants background removal
        remove_bg = input("Remove background? (y/n): ").lower().strip() == 'y'
        
        # Generate dynamic filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        directory = "image_bank/test"
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save original image
        output_name = os.path.join(directory, timestamp + ".png")
        cv2.imwrite(output_name, output)
        print(f"Original image saved as: {output_name}")
        
        # Save background-removed image if requested
        if remove_bg:
            bg_removed = remove_background(output)
            bg_removed_name = os.path.join(directory, timestamp + "_no_bg.png")
            bg_removed.save(bg_removed_name)
            print(f"Background-removed image saved as: {bg_removed_name}")
        
        # Save prompt to text file
        txt_name = os.path.join(directory, timestamp + ".txt")
        with open(txt_name, 'w') as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Negative Prompt: {neg_prompt}\n")
            f.write(f"Background Removed: {'Yes' if remove_bg else 'No'}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"Prompt saved as: {txt_name}")
