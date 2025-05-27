from PIL import Image

class imageSet:
    def __init__(self,path_to_dataset):
        self.images = []
        try:
            for img in path_to_dataset.iterdir():
                if img.is_file():
                    img = Image.open(img)
                    img = img.resize((50, 50)) 
                    self.images.append(img)
        except IOError:
            pass
        
    def print(self):
        for f in self.images:
            print(f)
        if self.images is None:
            print("Empty list")