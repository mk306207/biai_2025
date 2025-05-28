from neuralNetwork import neuralNetwork
from imageOperations import imageSet
from mockOneHot import meanColor, dominantColorLabel, onehot
from PIL import Image
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

def accuracy(y_true, y_pred):
    true_classes = np.argmax(y_true, axis=1)
    pred_classes = np.argmax(y_pred, axis=1)
    return np.mean(true_classes == pred_classes)


p = Path('data/')
pp = Path('test/')

i = imageSet(p)
test = imageSet(pp)
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

imagesArray = i.asArray(flatten=True)

n = neuralNetwork()
losses = n.train(imagesArray, onehot_y, epochs=1000)
epochs, loss_values = zip(*losses)

plt.figure(figsize=(10,5))
plt.plot(epochs, loss_values, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.grid(True)
plt.xticks(np.arange(min(epochs), max(epochs)+1, 100))
plt.show()

testArray = test.asArray(flatten=True)
Y_test = []
for image in test.images:
    mean_color = meanColor(image)
    label = dominantColorLabel(mean_color)

    Y_test.append(label)

Y_test = np.array(Y_test)
test_onehot_y = onehot(Y_test)
predictions = n.forwardPropagation(testArray)
acc = accuracy(test_onehot_y, predictions)
print(f"Accuracy: {acc*100:.2f}%")