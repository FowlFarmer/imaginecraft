from diffusers import StableDiffusionXLPipeline
import cv2
import numpy as np
import torch
import time
import os
import sys

other_dir = os.path.abspath("../backend")  # Change to actual path
sys.path.insert(0, other_dir)

import backend.sprites_pipeline as sprites_pipeline

def generate_dataset()