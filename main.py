from neuralNetwork import neuralNetwork
from imageOperations import imageSet
from pathlib import Path

p = Path('data/')


i = imageSet(p)
i.print()
#n = neuralNetwork()