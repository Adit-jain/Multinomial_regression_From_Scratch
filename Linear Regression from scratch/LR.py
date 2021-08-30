# Approximate Value
# y = wx + b

# cost function

# MSE = J(m,b) = 1/N * (for all i (actual-predicated)^2)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Linearregression:
    
    def __init__(self,filename,lr=0.001, iterations = 1000):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.filename = filename
        
        
    def plot_data(self,X,Y):
        plt.plot(X,Y)
        plt.scatter(X,Y,edgecolors='blue',color='red')
        plt.xlabel("Experience")
        plt.ylabel("Salary")
        plt.title("Experience vs Salaries")
        plt.show()
        
    def plot_reg_line(self,X,Y):
        plt.plot(X,(self.weights[0]*X)+self.bias)
        plt.scatter(X,Y,edgecolors='blue',color='red')
        plt.xlabel("Experience")
        plt.ylabel("Regression Line")
        plt.title("Regression Line")
        plt.show()
        
        
        
        
    def load_data(self):
        dataset = pd.read_csv(self.filename).values
        X=dataset[:,:-1]
        Y=dataset[:,-1]
        return X,Y
        
        
        
    def fit(self,X,Y):
        ##Initializing parameters
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        print("X_train_Shape : ", X.shape)
        print("Y_train_Shape : ", Y.shape)
        
        for _ in range(self.iterations):
            # New weight = Old weight - learning rate * derivative(old weight)
            # Y_predicted = wx+b
            
            
            Y_predicted = np.dot(X,self.weights) + self.bias
            
            dw = (1/n_samples) *(2* np.dot(X.T,(Y_predicted-Y)))
            db = (1/n_samples) *(2* np.sum(Y_predicted-Y))
            
            self.weights = self.weights - self.lr * dw
            self.bias = self.weights - self.lr * db
            
            
    
    
    def predict(self,X):
        Y_predicted = np.dot(X,self.weights) + self.bias
        return Y_predicted
    
    def print_MSE(self,Y_test,Y_pred):
        print("Mean Squared Error : ", np.sum(np.square(Y_pred-Y_test))/len(Y_pred))
    

obj = Linearregression("Salary_data.csv",lr=0.023,iterations=1000)
X,Y = obj.load_data()
obj.plot_data(X, Y)
obj.fit(X,Y)
Y_pred = obj.predict(X)
obj.plot_reg_line(X, Y)
obj.print_MSE(Y, Y_pred)
        
        