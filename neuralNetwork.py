import numpy as np
from matplotlib import pyplot as plt

class neuralNetwork:
    def __init__(self):
        try:
            self.learningRate = float(input("Please type in learning rate value (FLOAT!!!): "))
            self.b1 = np.zeros(1,32)
            self.w1 = np.random.randn(7500,32)*np.sqrt(2.0/7500) #He initialization for ReLU
            self.b2 = np.zeros(1,3)
            self.w2 = np.random.randn(32,3)*np.sqrt(1.0/32) # Xavier initialization for softmax
        except ValueError:
            print("Error - learning rate value should be a float datatype.")
            quit()
        # bias and weights, 3 layers: 1-input, 2-hidden layer, 3-output
        
    def ReLU(v):
        return np.maximum(0,v)