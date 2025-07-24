from PIL import Image
import numpy as np

class imageSet:
    def __init__(self,path_to_dataset):
        self.images = []
        try:
            for img in path_to_dataset.iterdir():
                if img.is_file():
                    img = Image.open(img)
                    img = img.resize((50, 50)) 
                    self.images.append(img)
                else:
                    print(f"Didnt load {img}")
        except IOError:
            print(f"Didnt load {img}")
        
    def print(self):
        for f in self.images:
            print(f)
        if self.images is None:
            print("Empty list")
    
    def asArray(self, flatten=True):
        result = []
        for img in self.images:
            arr = np.array(img) / 255.0
            if flatten:
                arr = arr.flatten()
            result.append(arr)
        return np.array(result)