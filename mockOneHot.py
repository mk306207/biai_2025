import numpy as np
import pandas as pd
from PIL import Image

def meanColor(image):
    img = image.copy().convert("RGB") #instead of RGB we must convert it into LAB(???)
    img = img.resize((64, 64))
    pixels = np.array(img)
    pixels = pixels.reshape(-1, 3)
    mean_color = np.mean(pixels, axis=0) / 255  
    mean_color = np.round(mean_color, 1)        
    return np.array(mean_color)

def dominantColorLabel(mean_color):
    return int(np.argmax(mean_color))

def onehot(y):
    return np.eye(3)[y]