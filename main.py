from neuralNetwork import neuralNetwork
from imageOperations import imageSet
from mockOneHot import meanColor, dominantColorLabel, onehot
from PIL import Image
from pathlib import Path
import numpy as np

p = Path('data/')


i = imageSet(p)
X = []
Y = []
for image in i.images:
    mean_color = meanColor(image)
    label = dominantColorLabel(mean_color)

    X.append(mean_color)
    Y.append(label)

X = np.array(X)
Y = np.array(Y)
onehot_y = onehot(Y)

# imagesArray = i.asArray()
# print(imagesArray.shape)
# n = neuralNetwork()
# n.forwardPropagation(imagesArray)

