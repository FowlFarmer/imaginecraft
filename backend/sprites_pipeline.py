from diffusers import DiffusionPipeline
import cv2
import numpy as np
import torch




if __name__ == "__main__":
    # to cuda
    pipe = DiffusionPipeline.from_pretrained("Raelina/Raehoshi-illust-XL-5.1").to("cuda")

    prompt = input("Enter your prompt: ")
    image = pipe(prompt).images[0]
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Generated Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()