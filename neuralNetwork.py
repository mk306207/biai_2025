import numpy as np


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
        return self.A2
        
    def crossEntropyLoss(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-9)) / m
    
    def backwardPropagation(self, X, y_true):
        m = X.shape[0]

        dZ2 = self.A2 - y_true #just a formula for gradient when using softmax + cross-entropy 
        dW2 = np.dot(self.A1.T, dZ2) / m #we calculate gradient for both weights and biases
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.w2.T)
        dZ1 = dA1 * self.ReLU_derivative(self.Z1) #same as above, but we must remember that we have called a ReLU func
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        self.w2 -= self.learningRate * dW2 #we update the values
        self.b2 -= self.learningRate * db2
        self.w1 -= self.learningRate * dW1
        self.b1 -= self.learningRate * db1
        
    def train(self, X, y_true, epochs=1000):
        losses = []
        for epoch in range(epochs):
            self.forwardPropagation(X)
            self.backwardPropagation(X, y_true)
            if epoch % 50 == 0:
                loss = self.crossEntropyLoss(y_true, self.A2)
                print(f"Epoch {epoch} - Loss: {loss:.4f}")
                losses.append((epoch, loss))
        return losses

    def predict(self, X):
        self.forwardPropagation(X)
        return np.argmax(self.A2, axis=1)
