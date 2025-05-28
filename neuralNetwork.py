import numpy as np
from matplotlib import pyplot as plt

class neuralNetwork:
    def __init__(self):
        try:
            self.learningRate = float(input("Please type in learning rate value (FLOAT!!!): "))
            self.b1 = None
            self.w1 = None
            self.b2 = None
            self.w2 = None
            self.b3 = None
            self.w3 = None
        except ValueError:
            print("Error - learning rate value should be a float datatype.")
            quit()
        # bias and weights, 3 layers: 1-input, 2-hidden layer, 3-output
        
    def ReLU(v):
        return np.maximum(0,v)