import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB # to compare
from sklearn import datasets
from helpers import *
from config import *
from random import randint

# for comparison
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

class MultipleRegression:

    w = [] # weights
    b = 0 # initial bias
    costs = []

    def fit(self, X, y):

        self.w = np.random.randn(X.shape[1]) # Initial random weights
        self.b = 0 # Initial bias
        self.w, self.b, self.costs = self.gradient_descent_function(X, y, self.w, self.b, epochs=2500)

    #cost function
    def cost_function(self, X, y, w, b):
        """
        Parameters:
        X: features
        y: target values
        w: weights
        b: bias
        
        Returns:
        cost: cost with current weights and bias
        """
        cost = np.sum((((X.dot(w) + b) - y) ** 2) / (2*len(y)))
        return cost

    # calculates the gradient descent
    def gradient_descent_function(self, X, y, w, b, alpha=0.01, epochs=1000):
        """
        Parameters:
        X: features
        y: target values
        w: initial weights
        b: initial bias
        alpha: learning rate
        epochs: number of iterations
        
        Returns:
        costs: cost per epoch
        w: finalised weights
        b: finalised bias
        """
        m = len(y)
        costs = [0] * epochs
        
        for epoch in range(epochs):
            # Calculate the value -- Forward Propagation
            z = X.dot(w) + b
            
            # Calculate the losses
            loss = z - y
            
            # Calculate gradient descent
            weight_gradient = X.T.dot(loss) / m
            bias_gradient = np.sum(loss) / m
            
            # Update weights and bias
            w = w - alpha*weight_gradient
            b = b - alpha*bias_gradient
            
            # Store current lost
            cost = self.cost_function(X, y, w, b)
            costs[epoch] = cost
            
        return w, b, costs

    # returns the score of the prediction
    def r2score(self, y_pred, y):
        """
        Parameters:
        y_pred: predicted values
        y: actual values
        
        Returns:
        r2: r2 score
        """
        rss = np.sum((y_pred - y) ** 2)
        tss = np.sum((y-y.mean()) ** 2)
        
        r2 = 1 - (rss / tss)
        return r2

    # returns the prediction
    def predict(self, X):
        return X.dot(self.w) + self.b

# testing
if __name__ == "__main__":

    state = random.randint(1, 999)

    for dataset in REGRESSION_DTS:

        if dataset == "concrete":
            df = parseDataset(dataset)

            #X = df.iloc[:, :-1]
            #y = df.iloc[:, -1:]

            X = df.drop(["Strength"], axis=1)
            y = df["Strength"]

            sc = StandardScaler()
            X = sc.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=state
            )

            mr = MultipleRegression()
            mr.fit(X_train, y_train)
            predictions = mr.predict(X_test) #y_predic

            #report =  classification_report(y_test, predictions)
            score = mr.r2score(y_test, predictions)
            #print("\nAlready implemented GNB accuracy:", accuracy(y_test, predictions))
            print("\nOur MR:\n")
            print("r2 score:", score)
            print("mean_sqrd_error is:", mean_squared_error(y_test, predictions))
            print("root_mean_squared error of is:", np.sqrt(mean_squared_error(y_test, predictions)))

            
            LR = LinearRegression() # creating an object of LinearRegression class
            LR.fit(X_train, y_train) # fitting the training data
            y_prediction =  LR.predict(X_test)

            # predicting the accuracy score
            score=r2_score(y_test,y_prediction)
            print("\nAlready implemented MR:\n")
            print("r2 score:", score)
            print("mean_sqrd_error:", mean_squared_error(y_test,y_prediction))
            print("root_mean_squared error of:", np.sqrt(mean_squared_error(y_test,y_prediction)))

        elif dataset == "abalone":
            df = parseDataset(dataset)

            #X = df.iloc[:, :-1]
            #y = df.iloc[:, -1:]

            X = df.drop(["age"], axis=1)
            y = df["age"]

            sc = StandardScaler()
            X = sc.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=state
            )

            mr = MultipleRegression()
            mr.fit(X_train, y_train)
            predictions = mr.predict(X_test) #y_predic

            #report =  classification_report(y_test, predictions)
            score = mr.r2score(y_test, predictions)
            #print("\nAlready implemented GNB accuracy:", accuracy(y_test, predictions))
            print("\nOur MR:\n")
            print("r2 score:", score)
            print("mean_sqrd_error is:", mean_squared_error(y_test, predictions))
            print("root_mean_squared error of is:", np.sqrt(mean_squared_error(y_test, predictions)))

            
            LR = LinearRegression() # creating an object of LinearRegression class
            LR.fit(X_train, y_train) # fitting the training data
            y_prediction =  LR.predict(X_test)

            # predicting the accuracy score
            score=r2_score(y_test,y_prediction)
            print("\nAlready implemented MR:\n")
            print("r2 score:", score)
            print("mean_sqrd_error:", mean_squared_error(y_test,y_prediction))
            print("root_mean_squared error of:", np.sqrt(mean_squared_error(y_test,y_prediction)))