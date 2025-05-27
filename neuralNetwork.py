import numpy as np
from matplotlib import pyplot as plt

class neuralNetwork:
    def __init__(self):
        try:
            learningRate = float(input("Please type in learning rate value (FLOAT!!!)"))
        except ValueError:
            print("Error - learning rate value should be a float datatype.")
            quit()
        
    def ReLU(v):
        return np.maximum(0,v)