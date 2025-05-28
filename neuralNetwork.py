import numpy as np
from matplotlib import pyplot as plt

class neuralNetwork:
    def __init__(self):
        try:
            self.learningRate = float(input("Please type in learning rate value (FLOAT!!!): "))
            self.b1 = np.zeros((1,32))
            self.w1 = np.random.randn(7500,32)*np.sqrt(2.0/7500) #He initialization for ReLU
            self.b2 = np.zeros((1,3))
            self.w2 = np.random.randn(32,3)*np.sqrt(1.0/32) # Xavier initialization for softmax
        except ValueError:
            print("Error - learning rate value should be a float datatype.")
            quit()
        # bias and weights, 3 layers: 1-input, 2-hidden layer, 3-output
        
    def ReLU(self,v):
        return np.maximum(0,v)
    def ReLU_derivative(self,v):
        return (v > 0).astype(float)
    def softmax(self, v):
        e_v = np.exp(v - np.max(v, axis=1, keepdims=True))
        return e_v / e_v.sum(axis=1, keepdims=True)
    
    def forwardPropagation(self, X):
        self.Z1 = np.dot(X, self.w1) + self.b1 # Matrix X - our raw matrix of RGB values is multiplied by w1. X.shape = (86,7500), w1.shape = (7500,32), which will give us Z1.shape = (86,32)
        self.A1 = self.ReLU(self.Z1)
        self.Z2 = np.dot(self.A1, self.w2) + self.b2
        self.A2 = self.softmax(self.Z2)
        print(self.A2)
        
        